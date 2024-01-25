import torch
import torch.nn.functional as F
from imi_image_warp import compute_identity_grid

def compute_jacobian_det(field, grid=None, spacing=(1, 1, 1)):
    if field.dim() == 4:
        return compute_jacobian_det_2d(field, grid)
    else:
        return compute_jacobian_det_3d(field, grid)


def _determinant_2x2(a00, a01, a10, a11):
    return a00 * a11 - a01 * a10  # compute determinant of 2x2 matrix


def compute_jacobian_det_2d(field, grid=None, spacing=(1, 1, 1)):
    assert field.dim() == 4  # assume 2D displacement field
    assert field.shape[1] == 2  # assume 2D displacement field in (N, 2, H, W)
    if grid is None:
        grid = compute_identity_grid(field.shape, device=field.device, dtype=field.dtype)
    # compute spacing in pytorch grid
    # spacing = (grid[0, 0, 1, 1] - grid[0, 0, 1, 0], grid[0, 1, 1, 1] - grid[0, 1, 0, 1])
    spacing = (1. / (field.shape[-1]/2), 1. / (field.shape[-2]/2))
    # compute transform - not displacement
    warp_transf = field + grid
    # use forward differences
    #dx = (warp_transf[:, :, :-1, 1:] - warp_transf[:, :, :-1, :-1]) / spacing[0]  # divide by spacing for finite differences
    #dy = (warp_transf[:, :, 1:, :-1] - warp_transf[:, :, :-1, :-1]) / spacing[1]
    # use central differences (same as ITK)
    dx = (warp_transf[:, :, 1:-1, 2:] - warp_transf[:, :, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (warp_transf[:, :, 2:, 1:-1] - warp_transf[:, :, :-2, 1:-1]) / (2 * spacing[1])
    J00, J01 = dx[:, 0, :, :], dx[:, 1, :, :]
    J10, J11 = dy[:, 0, :, :], dy[:, 1, :, :]
    Jdet = _determinant_2x2(J00, J01, J10, J11)  # compute determinant of 2x2 matrix
    return F.pad(Jdet, (1, 1, 1, 1), mode="constant", value=1.0)  # change padding to (0,1,0,1) for forward differences


def compute_jacobian_det_3d(field, grid=None, spacing=(1, 1, 1)):
    assert field.dim() == 5  # assume 3D displacement field
    assert field.shape[1] == 3  # assume 3D displacement field in (N, 2, D, H, W)
    if grid is None:
        grid = compute_identity_grid(field.shape, device=field.device, dtype=field.dtype)
    # compute spacing in pytorch grid
    # spacing = (grid[0, 0, 1, 1] - grid[0, 0, 1, 0], grid[0, 1, 1, 1] - grid[0, 1, 0, 1])
    spacing = (1. / (field.shape[-1]/2), 1. / (field.shape[-2]/2), 1. / (field.shape[-3]/2))
    # compute transform - not displacement
    warp_transf = field + grid
    # use central differences (same as ITK)
    dx = (warp_transf[:, :, 1:-1, 1:-1, 2:] - warp_transf[:, :, 1:-1, 1:-1, :-2]) / (2 * spacing[0])  # divide by spacing for finite differences
    dy = (warp_transf[:, :, 1:-1, 2:, 1:-1] - warp_transf[:, :, 1:-1, :-2, 1:-1]) / (2 * spacing[1])
    dz = (warp_transf[:, :, 2:, 1:-1, 1:-1] - warp_transf[:, :, :-2, 1:-1, 1:-1]) / (2 * spacing[2])
    J00, J01 , J02 = dx[:, 0, :, :, :], dx[:, 1, :, :, :], dx[:, 2, :, :, :]
    J10, J11 , J12 = dy[:, 0, :, :, :], dy[:, 1, :, :, :], dy[:, 2, :, :, :]
    J20, J21 , J22 = dz[:, 0, :, :, :], dz[:, 1, :, :, :], dz[:, 2, :, :, :]
    Jdet = J00 * _determinant_2x2(J11, J12, J21, J22) \
           - J01 * _determinant_2x2(J10, J12, J20, J22) \
           + J02 * _determinant_2x2(J10, J11, J20, J21)  # compute determinant of 3x3 matrix
    return F.pad(Jdet, (1, 1, 1, 1, 1, 1), mode="constant", value=1.0)  # change padding to (0,1,0,1) for forward differences
