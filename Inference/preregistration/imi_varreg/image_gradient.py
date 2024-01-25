import torch
import torch.nn.functional as F


def image_gradient_central_2d(image, spacing=(1, 1)):
    assert image.dim() == 4  # assume 2D image NxCxHxW
    assert image.shape[1] == 1  # assume 2D gray image (N, 1, H, W)
    assert len(spacing) == 2
    # use forward differences
    #dx = (warp_transf[:, :, :-1, 1:] - warp_transf[:, :, :-1, :-1]) / spacing[0]  # divide by spacing for finite differences
    #dy = (warp_transf[:, :, 1:, :-1] - warp_transf[:, :, :-1, :-1]) / spacing[1]
    # use central differences (same as ITK)
    dx = (image[:, :, 1:-1, 2:] - image[:, :, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (image[:, :, 2:, 1:-1] - image[:, :, :-2, 1:-1]) / (2 * spacing[1])
    gradient = torch.cat((dx, dy), dim=1)
    return F.pad(gradient, (1, 1, 1, 1), mode="constant", value=0.0)


def image_gradient_central_3d(image, spacing=(1, 1, 1)):
    assert image.dim() == 5  # assume 3D image NxCxDxHxW
    assert image.shape[1] == 1  # assume 3D gray image (N, 1, D, H, W)
    assert len(spacing) == 3
    # use central differences (same as ITK)
    dx = (image[:, :, 1:-1, 1:-1, 2:] - image[:, :, 1:-1, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (image[:, :, 1:-1, 2:, 1:-1] - image[:, :, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    dz = (image[:, :, 2:, 1:-1, 1:-1] - image[:, :, :-2, 1:-1, 1:-1]) / (2 * spacing[2])
    gradient = torch.cat((dx, dy, dz), dim=1)
    return F.pad(gradient, (1, 1, 1, 1, 1, 1), mode="constant", value=0.0)
