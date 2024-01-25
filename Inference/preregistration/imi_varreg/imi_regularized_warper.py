import torch
from torch.autograd import Function
from preregistration.imi_varreg.imi_regularize_diffusive import ImiRegularizeDiffusive
from preregistration.imi_varreg.imi_deformation import ImiDisplacementDeformation, ImiVelocityDeformation
from preregistration.imi_varreg.image_gradient import image_gradient_central_3d, image_gradient_central_2d
# DEBUG
from preregistration.imi_varreg.imi_debug_tools import debug_write_image_sitk, debug_print, debug_print_field_statistics


class ImiRegularizedWarper(torch.nn.Module):
    def __init__(self, img_size, alpha, tau=1.0, spacing=(1, 1, 1), regularizer='diffusive',
                 gradient_scaling='none', diffeomorphic=False, loss_reduction_type='mean', use_spacing=True):
        super().__init__()
        if regularizer == 'diffusive':
            self.regularizer = ImiRegularizeDiffusive(img_size, alpha, tau, spacing, use_spacing)
        else:
            raise NotImplementedError(f"Unknown regularizer type '{regularizer}', supported regularizers: 'diffusive'.")
        if not diffeomorphic:
            self.warper = ImiDisplacementDeformation(img_size, spacing)
        else:
            self.warper = ImiVelocityDeformation(img_size, spacing)
        if gradient_scaling == 'none':
            self.compute_spatial_gradient_func = self.compute_spatial_gradient_scale_none
        elif gradient_scaling == 'demons':
            self.compute_spatial_gradient_func = self.compute_spatial_gradient_scale_demons
        elif gradient_scaling == 'spacing':
            self.compute_spatial_gradient_func = self.compute_spatial_gradient_scale_sqr_spacing
        elif gradient_scaling == 'normalize':
            self.compute_spatial_gradient_func = self.compute_spatial_gradient_scale_normalize
        else:
            raise NotImplementedError(f"Unknown gradient scaling '{gradient_scaling}', supported gradient "
                                      f"scaling: 'none'|'demons'|'spacing'|'normalize'.")

        self.dim = len(img_size) - 2
        self.numpix = torch.prod(torch.tensor(img_size[2:]))
        self.spacing = spacing
        assert self.dim == 2 or self.dim == 3
        assert len(spacing) == self.dim
        self.alpha = alpha
        self.tau = tau
        self.speed_value_threshold = 0.00001
        self.denominator_threshold = 0.0001
        self.mean_squared_spacing = sum([s*s for s in spacing]) / len(spacing)
        self.image_gradient_func = image_gradient_central_3d if len(spacing) == 3 else image_gradient_central_2d

        self.forward_backward_functional = ImiRegularizedWarperForwardBackwardFunctional

    def forward(self, image, displacement):
        args = [image, displacement] + [self]
        return self.forward_backward_functional.apply(*args)

    def apply_forward(self, image, displacement):
        with torch.no_grad():
            warped_image = self.warper(image, displacement)
            regularizer_loss = self.regularizer.compute_loss(displacement)
        return warped_image.requires_grad_(True), regularizer_loss

    def update_displacement(self, speed_image, warped_image, input_displacement):
        with torch.no_grad():
            debug_print_field_statistics(7, input_displacement, "input_displacement")
            update_field = self.compute_spatial_gradient_func(speed_image, warped_image)
            debug_print_field_statistics(7, update_field, "update_field")
            debug_write_image_sitk(9, update_field, "pyreg_update.nii.gz")
            regularized_displacement = self.regularizer(input_displacement - self.tau * update_field)
            debug_print_field_statistics(7, regularized_displacement, "regularized_displacement")
        return regularized_displacement

    def compute_spatial_gradient_scale_demons(self, speed_image, input_image):
        speed_image = speed_image * self.numpix
        debug_print_field_statistics(7, speed_image, "speed_image")
        image_gradient = self.image_gradient_func(input_image, spacing=self.spacing)
        debug_print_field_statistics(7, image_gradient, "image_gradient")
        denominator = torch.square(speed_image) / self.mean_squared_spacing \
                          + torch.sum(torch.square(image_gradient), dim=1, keepdim=True)
        zero_mask = torch.logical_or((speed_image.abs() < self.speed_value_threshold),
                                     (denominator < self.denominator_threshold))
        image_gradient[zero_mask.expand(image_gradient.size())] = 0.0
        mask = torch.logical_not(zero_mask).expand(image_gradient.size())
        image_gradient[mask] *= (speed_image.expand(image_gradient.size())[mask]
                                 / denominator.expand(image_gradient.size())[mask])
        return image_gradient

    def compute_spatial_gradient_scale_sqr_spacing(self, speed_image, input_image):
        speed_image = speed_image * self.numpix
        debug_print_field_statistics(7, speed_image, "speed_image")
        debug_write_image_sitk(9, speed_image, "pyreg_speedimage.nii.gz")
        image_gradient = self.image_gradient_func(input_image, spacing=self.spacing)
        debug_print_field_statistics(7, image_gradient, "image_gradient")
        image_gradient *= speed_image * self.mean_squared_spacing
        return image_gradient

    def compute_spatial_gradient_scale_none(self, speed_image, input_image):
        speed_image = speed_image * self.numpix
        debug_print_field_statistics(7, speed_image, "speed_image")
        image_gradient = self.image_gradient_func(input_image, spacing=self.spacing)
        debug_print_field_statistics(7, image_gradient, "image_gradient")
        image_gradient *= speed_image
        return image_gradient

    def compute_spatial_gradient_scale_normalize(self, speed_image, input_image):
        speed_image = 0.5*speed_image * self.numpix
        debug_print_field_statistics(7, speed_image, "speed_image")
        image_gradient = self.image_gradient_func(input_image, spacing=self.spacing)
        debug_print_field_statistics(7, image_gradient, "image_gradient")
        mean_squared_diff = speed_image.pow(2).mean()
        mean_squared_grad_magnitude = image_gradient.pow(2).sum(dim=1).mean()
        factor = 1.0 / (mean_squared_diff/self.mean_squared_spacing + mean_squared_grad_magnitude)
        debug_print(7, f"factor={factor}, ({mean_squared_diff}/{self.mean_squared_spacing} + {mean_squared_grad_magnitude})")
        image_gradient *= speed_image * factor
        return image_gradient


class ImiRegularizedWarperForwardBackwardFunctional(Function):
    """
    forward/backward functional that is used in ImplicitDiffusion module.
    """
    @staticmethod
    def forward(ctx, *args):
        ctx.warper_object = args[-1]
        input_image = args[0]
        input_displacement = args[1]
        # compute diffused version
        out_warped, regularizer_loss = ctx.warper_object.apply_forward(input_image, input_displacement)
        ctx.save_for_backward(out_warped, input_displacement)
        return out_warped, regularizer_loss

    @staticmethod
    def backward(ctx, grad_output, grad_loss_output):
        warped_image = ctx.saved_tensors[0]
        input_displacement = ctx.saved_tensors[1]
        regularized_displacement = ctx.warper_object.update_displacement(grad_output, warped_image, input_displacement)
        return None, input_displacement - regularized_displacement, None
