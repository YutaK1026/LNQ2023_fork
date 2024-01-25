import torch
from image_gradient import image_gradient_central_3d


class ImiThirionLoss(torch.nn.Module):
    def __init__(self, img_dim=3, spacing=(1, 1, 1)):
        super().__init__()
        assert img_dim == len(spacing)
        self.speed_value_threshold = 0.001
        self.denominator_threshold = 0.001
        self.spacing = spacing
        self.mean_squared_spacing = sum([s*s for s in spacing]) / img_dim

    def compute_gradient(self, fixed_image, moving_image):
        speed_value = moving_image - fixed_image
        speed_value[speed_value.abs() < self.speed_value_threshold] = 0.0
        moving_gradient = image_gradient_central_3d(moving_image, spacing=self.spacing)
        denominator = torch.square(speed_value) / self.mean_squared_spacing \
                      + torch.sum(torch.square(moving_gradient), dim=1, keepdim=True)
        zero_mask = torch.logical_or((speed_value.abs() < self.speed_value_threshold),
                                     (denominator < self.denominator_threshold))
        moving_gradient[zero_mask.expand(moving_gradient.size())] = 0.0
        mask = torch.logical_not(zero_mask).expand(moving_gradient.size())
        moving_gradient[mask] *= (speed_value.expand(moving_gradient.size())[mask] / denominator.expand(moving_gradient.size())[mask])
        return moving_gradient

    def compute_loss(self, fixed_image, moving_image):
        return 0.5 * torch.square(moving_image - fixed_image)


if __name__ == '__main__':
    import argparse
    from imi_image_warp import load_image_sitk, sitk_image_to_tensor, tensor_image_to_sitk, write_image_sitk

    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', '-f', required=True, type=str, help='path for input image')
    parser.add_argument('--moving', '-m', required=True, type=str, help='path for input image')
    parser.add_argument('--output', '-o', required=True, type=str, help='path for output image')
    args = parser.parse_args()

    fixed_image, fixed_meta = sitk_image_to_tensor(load_image_sitk(args.fixed), return_meta_data=True)
    moving_image, moving_meta = sitk_image_to_tensor(load_image_sitk(args.moving), return_meta_data=True)

    loss = ImiThirionLoss(img_dim=3, spacing=fixed_meta['spacing'])
    loss_value = loss.compute_loss(fixed_image, moving_image).mean()
    loss_grad = loss.compute_gradient(fixed_image, moving_image)
    write_image_sitk(tensor_image_to_sitk(loss_grad, fixed_meta), args.output)
