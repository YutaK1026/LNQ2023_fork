import numpy as np
from time import time
import torch
import sys
import os
import glob
from pathlib import Path
from natsort import natsorted
from torch import autocast
from nnUnet.nnUNetTrainerSmartLoadBougetSampling import nnUNetTrainerSmartLoadBougetSampling
from nnUnet.polylr_refinement import PolyLRScheduler
from nnunetv2.utilities.helpers import dummy_context
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.utilities.helpers import empty_cache
from nnUnet.rand_conv3d import GINGroupConv3D
from nnUnet.adv_bias import AdvBias, rescale_intensity
from torch.nn.functional import interpolate
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss
from torch import nn
from monai.losses import TverskyLoss
from monai.transforms import Compose, RandGaussianNoise, RandAdjustContrast
from nnUnet.probmap_weighted_diceloss_utils import DC_and_CE_loss, MemoryEfficientSoftDiceLoss, create_weight_map


def normalize(input):
    output = (input - torch.mean(input)) / torch.std(input)
    return output


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return np.exp(-5.0 * phase * phase).astype(float)


def get_current_weight(epoch, saturation_epoch, max_weight=1):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return max_weight * sigmoid_rampup(epoch, saturation_epoch)


class modifiedLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.diceceloss = loss
        self.tverskyloss = TverskyLoss(softmax=True, to_onehot_y=True, include_background=False, alpha=0.25, beta=0.75)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, weight_map: torch.Tensor = None):
        diceceloss = self.diceceloss(net_output, target, weight_map)
        tverskyloss = self.tverskyloss(net_output, target)
        return diceceloss + tverskyloss


class trainer(nnUNetTrainerSmartLoadBougetSampling):

    def __init__(self, plans, configuration, fold, dataset_json):
        super().__init__(plans=plans, configuration=configuration, fold=fold, dataset_json=dataset_json)
        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-2
        self.weight_decay = 3e-5
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 75
        self.num_batches_cached = 10
        self.num_val_iterations_per_epoch = 20
        self.replace_rate = 0.1  # switch all in 10 epochs
        self.current_epoch = 0
        self.num_epochs = plans['num_epochs']
        self.use_random_conv_until = plans['use_random_conv_until']
        self.use_alpha_ramp = True
        ###
        self.augmenter = GINGroupConv3D(in_channel=1, out_channel=1, interm_channel=4, n_layer=4).to(self.device, non_blocking=True)
        blender_cofig = {
            'epsilon': 0.3,
            'xi': 1e-6,
            'control_point_spacing': [24] * 3,
            'downscale': 2,
            'data_size': [2, 1, 56, 128, 224],
            'interpolation_order': 2,
            'init_mode': 'gaussian',
            'space': 'log'
        }
        self.blender = AdvBias(config_dict=blender_cofig, use_gpu=True)
        self.blender.init_parameters()
        ###
        self.transforms = Compose(
            [
                RandGaussianNoise(prob=0.5, mean=0, std=0.1),
                RandAdjustContrast(prob=0.5)
            ]
        )
        self.saturation_epoch = plans['saturation_epoch']
        self.batch_size = max(1, self.batch_size // 2)  # batch size is reduced due to sampling from two dataloaders
        self.max_steps_lr_decay = plans['max_epochs_lr_decay']
        self.print_to_log_file(f"Current learning rate: {self.initial_lr}, epochs: {self.num_epochs}, lr-decay for {self.max_steps_lr_decay} epochs")
        self.print_to_log_file(f"Training   Iter per epoch (cache size): {self.num_iterations_per_epoch}, batches chached (DA processes): {self.num_batches_cached}")
        self.print_to_log_file(f"Validation Iter per epoch (cache size): {self.num_val_iterations_per_epoch}")
        if self.num_epochs <= self.use_random_conv_until:
            use_rand_convs = True
        else:
            use_rand_convs = False
        self.print_to_log_file(f"Use random convolutions: {use_rand_convs} with alpha ramp: {self.use_alpha_ramp} for {self.saturation_epoch} epochs " if use_rand_convs else "")

    def initialize(self):
        super().initialize()
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}, {self.initial_lr}, {self.num_epochs}")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.max_steps_lr_decay, self.num_epochs)
        return optimizer, lr_scheduler

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

        loss = modifiedLoss(loss)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        # Create probmap based weight map
        deep_supervision_scales = self._get_deep_supervision_scales()
        weight_mask = create_weight_map(data.clone(), [i.clone() for i in target], deep_supervision_scales)
        weight_mask = [i.to(self.device, non_blocking=True) for i in weight_mask]

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        data[:, 0] = self.transforms(data[:, 0].unsqueeze(dim=1)).squeeze(dim=1)

        if self.current_epoch < self.use_random_conv_until:
            if self.use_alpha_ramp:
                self.augmenter.adjust_alpha_max(get_current_weight(self.current_epoch, self.saturation_epoch))
            else:
                self.augmenter(1.0)

            # set blender config
            self.blender.init_config({
                'epsilon': 0.3,
                'xi': 1e-6,
                'control_point_spacing': [24] * 3,
                'downscale': 2,  #
                'data_size': list(data[:, 0, None,...].shape),
                'interpolation_order': 2,
                'init_mode': 'gaussian',
                'space': 'log'
            })
            # get bias field
            self.blender.reset_bias_value()
            # TODO: FIX B-spline interpolation in adv_bias.py --> outputs wrong tensor size!
            blend_mask = rescale_intensity(interpolate(self.blender.bias_field,size=data[0, 0,...].shape, mode='trilinear'))
            # compute two random augmentations
            data_aug1 = rescale_intensity(self.augmenter(data[:, 0].unsqueeze(dim=1)))
            data_aug2 = rescale_intensity(self.augmenter(data[:, 0].unsqueeze(dim=1)))
            # blending of both augmentations
            data_blend = data_aug1.clone().detach() * blend_mask + data_aug2.clone().detach() * (1 - blend_mask)
            data[:,0] = normalize(data_blend)[:,0]

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # for idx in range(len(output)):
            #    print(f"pred {idx} [{torch.min(output[idx]).item()} - {torch.max(output[idx]).item()}] tar [{torch.min(target[idx]).item()} - {torch.max(target[idx]).item()}]")
            # del data
            l = self.loss(output, target, weight_mask)
            # Compute FPR and FNR
            p0, g0 = torch.nn.functional.softmax(output[0], dim=1)[:, 1:].detach(), target[0][:].detach()
            p1, g1 = 1 - p0, 1 - g0
            tp = torch.sum(p0 * g0, (2, 3, 4))
            tn = torch.sum(p1 * g1, (2, 3, 4))
            fp = torch.sum(p0 * g1, (2, 3, 4))
            fn = torch.sum(p1 * g0, (2, 3, 4))
            fnr = (fn + 1e-5) / (fn + tp + 1e-5)
            fpr = (fp + 1e-5) / (fp + tn + 1e-5)
            # print(torch.mean(fnr).item(), torch.mean(fpr).item())
            # print(l)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(), 'FNR': torch.mean(fnr).item(), 'FPR': torch.mean(fpr).item()}

    def run_training(self):

        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_data = {}
                patch_train, patch_extra = next(self.dataloader_train), next(self.dataloader_train_extra)
                train_data['data'] = torch.cat((patch_train['data'], patch_extra['data']), dim=0)
                train_data['target'] = [torch.cat((patch_train_target, patch_extra_target), dim=0)
                                        for patch_train_target, patch_extra_target in
                                        zip(patch_train['target'], patch_extra['target'])]
                train_outputs.append(self.train_step(train_data))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()

        self.on_train_end()

    def on_train_end(self):
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        # # now we can delete latest
        # if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
        #     os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None:
                self.dataloader_train.finish()
                self.dataloader_train_extra.finish()
            if self.dataloader_val is not None:
                self.dataloader_val.finish()
            sys.stdout = old_stdout

        empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def get_checkpoint(self, save_dir='models', best=True):
        file_names = [f for f in os.listdir(os.path.join(self.output_folder, save_dir)) if f.endswith('.pth')]
        sorted_file_names = natsorted(file_names, key=lambda f: f[f.find('dsc'):])
        best_model = sorted_file_names[-1] if best else sorted_file_names[0]
        epoch = int(best_model[best_model.find('epoch') + 5:][:4]) if best_model.find('epoch') >= 0 else 100  # default epoch
        return os.path.join(self.output_folder, save_dir, best_model), epoch

    def save_n_best_checkpoints(self, save_dir='models', max_model_num=8):
        dsc = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
        filename = f'checkpoint_epoch{self.current_epoch:04d}_dsc{np.round(dsc, decimals=4)}.pth'
        Path(join(self.output_folder, save_dir)).mkdir(exist_ok=True)
        model_lists = natsorted(glob.glob(join(self.output_folder, save_dir, '*')))
        if len(model_lists) > max_model_num:
            last_filename, epoch = self.get_checkpoint(save_dir, best=False)
            print(f'remove chk-pt file {os.path.basename(last_filename)} from epoch {epoch}.')
            os.remove(last_filename)
        self.save_checkpoint(join(self.output_folder, save_dir, filename))

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        # todo find a solution for this stupid shit
        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))
            # Save n-best checkpoints
            self.save_n_best_checkpoints()

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=6)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)
        # train dataloader is smart-cache and we have to update the cache
        self.dataloader_train.update_cache()
        self.dataloader_train_extra.update_cache()