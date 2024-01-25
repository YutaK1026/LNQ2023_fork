import torch
import torch.nn.functional as F
import numpy as np
import math
# DEBUG
from preregistration.imi_varreg.imi_debug_tools import debug_write_image_sitk


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [5] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(device=Ii.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        win_size = np.prod(win)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        IvarJvar = I_var * J_var

        #cc = (cross * cross) / (I_var * J_var + 1e-4)
        cc = (cross * cross + 1e-5) / (IvarJvar + 1e-5)
        #cc[IvarJvar < 1e-5] = 0

        """
        I_mean = conv_fn(Ii, sum_filt, stride=stride, padding=padding) / win_size
        J_mean = conv_fn(Ji, sum_filt, stride=stride, padding=padding) / win_size
        I_center = Ii - I_mean
        J_center = Ji - J_mean
        I2 = I_center * I_center
        J2 = J_center * J_center
        IJ = I_center * J_center
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)
        IvarJvar = I2_sum.sqrt() * J2_sum.sqrt()
        cc = (I_center * J_center) / (IvarJvar + 1e-5)
        #cc[IvarJvar < 1e-5] = 0
        """
        debug_write_image_sitk(9, cc.detach(), "pyreg_ncc_value.nii.gz")
        return 1.0 - torch.mean(cc)

