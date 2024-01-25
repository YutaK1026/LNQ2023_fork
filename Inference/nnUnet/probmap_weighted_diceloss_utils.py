import numpy as np
from time import time
import SimpleITK as sitk
import torch
from torch import nn
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
# compund_losses
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from typing import Callable
# dice
from nnunetv2.utilities.ddp_allgather import AllGatherGrad



class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None, weight_map: torch.Tensor = None):
        shp_x, shp_y = x.shape, y.shape

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]

        # make everything shape (b, c)
        axes = list(range(2, len(shp_x)))

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(shp_x, shp_y)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                gt = y.long()
                y_onehot = torch.zeros(shp_x, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, gt, 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        if weight_map is not None:
            intersect = x * y_onehot if loss_mask is None else x * y_onehot * loss_mask
            sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

            if self.batch_dice:
                sum_pred = sum_pred.sum(0)
                sum_gt = sum_gt.sum(0)

            dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))
            dc = (dc * weight_map).sum(axes).sum(0)

            dc = dc.mean()

        else:
            intersect = (x * y_onehot).sum(axes) if loss_mask is None else (x * y_onehot * loss_mask).sum(axes)
            sum_pred = x.sum(axes) if loss_mask is None else (x * loss_mask).sum(axes)

            if self.ddp and self.batch_dice:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            if self.batch_dice:
                intersect = intersect.sum(0)
                sum_pred = sum_pred.sum(0)
                sum_gt = sum_gt.sum(0)

            dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

            dc = dc.mean()
        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor = None):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask, weight_map=weight_map) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


def create_weight_map(data: list, target: list, deep_supervision_scales: list):
    weight_mask = [torch.ones_like(i) for i in target]
    gt_segm = [torch_dilate(i, 2) for i in target]
    ds_transform = DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='probm', output_key='probm')
    prob_mask = ds_transform(**{'probm': data[:, -1].unsqueeze(dim=1).detach().cpu().numpy()})['probm']
    prob_mask = [torch.from_numpy(i) for i in prob_mask]
    # sitk.WriteImage(sitk.GetImageFromArray(prob_mask[0][0].squeeze().cpu().numpy()),
    #                 '/share/data_rosita1/engelson/Projects/Lnq2023/results/prob_mask.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(gt_segm[0][0].squeeze().cpu().numpy()),
    #                 '/share/data_rosita1/engelson/Projects/Lnq2023/results/gt_segm.nii.gz')
    for i in range(len(weight_mask)):
        weight_mask[i] = 1 - prob_mask[i]
        weight_mask[i][gt_segm[i] > 0] = 1
        weight_mask[i] = torch.clip(weight_mask[i], 0.75, 1)
        break
    return weight_mask


def dilate_rad2(img: torch.Tensor):
    batch_size = img.shape[0]
    for i in range(batch_size):
        img_sitk = sitk.DilateObjectMorphology(sitk.GetImageFromArray(img[i].squeeze().long().cpu().numpy()),
                                               kernelRadius=(2, 2, 2))
        img_dilated = torch.from_numpy(sitk.GetArrayFromImage(img_sitk)).unsqueeze(0).unsqueeze(0)
        img[i] = img_dilated
    return img

def torch_dilate(img: torch.Tensor, radius: int=2):
    kernel_size = radius * 2 + 1
    padding = radius
    dilated_image = nn.functional.max_pool3d(img, kernel_size, stride=1, padding=padding)
    return dilated_image
