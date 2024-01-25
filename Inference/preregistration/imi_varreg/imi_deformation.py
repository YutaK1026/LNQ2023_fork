import torch
import torch.nn.functional as F
import SimpleITK as sitk


class ImiDisplacementDeformation(torch.nn.Module):
    def __init__(self, img_size, spacing=(1, 1, 1), mode='bilinear', padding_mode='border'):
        super().__init__()
        assert len(img_size) - 2 == len(spacing)
        self.img_size = img_size
        self.spacing = spacing
        self.mode = mode
        self.padding_mode = padding_mode
        pytorch_grid = compute_identity_grid_torch(img_size)
        self.register_buffer('pytorch_grid', pytorch_grid)

    def forward(self, image, displacement_field):
        return warp_image_with_displacement_field(image, displacement_field, self.pytorch_grid, self.spacing,
                                                  self.mode, self.padding_mode)


class ImiVelocityDeformation(ImiDisplacementDeformation):
    def __init__(self, img_size, spacing=(1, 1, 1), accuracy=7, mode='bilinear', padding_mode='border'):
        super().__init__(img_size, spacing, mode, padding_mode)
        self.accuracy = accuracy

    def forward(self, image, velocity_field):
        return warp_image_with_velocity_field(image, velocity_field, self.pytorch_grid, self.spacing,
                                              self.accuracy, self.mode, self.padding_mode)


#
#         FUNCTIONS
#
def warp_image_with_displacement_field(image, displ_field, pytorch_grid=None, spacing=(1, 1, 1), mode='bilinear', padding_mode='zeros'):
    # DEBUG
    assert isinstance(image, torch.Tensor)
    assert isinstance(displ_field, torch.Tensor)
    # convert displacement field from world space to torch space
    pytorch_field = convert_field_from_world_to_torch_space(displ_field, spacing)
    if pytorch_grid is None:
        pytorch_grid = compute_identity_grid_torch(displ_field.shape, device=displ_field.device, dtype=displ_field.dtype)
    mapping_field = _convert_to_channel_last(pytorch_grid + pytorch_field)  # move components to last position
    out = F.grid_sample(image, mapping_field, mode=mode, padding_mode=padding_mode, align_corners=False)
    return out


def warp_image_with_velocity_field(input_image, velo_field, pytorch_grid=None, spacing=(1, 1, 1), accuracy=7, mode='bilinear', padding_mode='zeros'):
    if pytorch_grid is None:
        pytorch_grid = compute_identity_grid_torch(velo_field.shape, device=velo_field.device, dtype=velo_field.dtype)

    displ_field = scaling_and_squaring(velo_field, pytorch_grid, spacing, accuracy, mode, padding_mode)
    return warp_image_with_displacement_field(input_image, displ_field, pytorch_grid, spacing, mode, padding_mode)


def scaling_and_squaring(velocity, pytorch_grid=None, spacing=(1, 1, 1), accuracy=7, mode='bilinear', padding_mode='border'):
    """ Implements the scaling and squaring algorithm to compute a displacement field from a velocity field."""
    if pytorch_grid is None:
        pytorch_grid = compute_identity_grid_torch(velocity.shape, device=velocity.device, dtype=velocity.dtype)

    velocity = convert_field_from_world_to_torch_space(velocity / (2 ** accuracy), spacing=spacing)
    for i in range(accuracy):
        mapping_field = _convert_to_channel_last(velocity + pytorch_grid)  # move components to last position
        velocity = velocity + F.grid_sample(velocity, mapping_field,
                                            mode=mode, padding_mode=padding_mode, align_corners=False)
    return convert_field_from_torch_to_world_space(velocity, spacing=spacing)


@torch.no_grad()
def compute_identity_grid_torch(size, device=None, dtype=None):
    # find the right set of parameters
    dim = len(size) - 2  # assume NxCxHxW or NxCxDxHxW
    batch_dim = size[0]
    if dim == 2:  # 2D field
        id_theta = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]], dtype=dtype, device=device)
    elif dim == 3:  # 3D field
        id_theta = torch.tensor([[[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]], dtype=dtype, device=device)
    else:
        raise ValueError('Dimension of identity field is {}, expected is 4D or 5D (NxCxHxW or NxCxDxHxW)'.format(len(size)))
    id_theta = id_theta.expand(batch_dim, *id_theta.shape[1:])
    identity_grid = F.affine_grid(id_theta, size, align_corners=False)
    return _convert_from_channel_last(identity_grid)  # move the components to 2nd position


@torch.no_grad()
def identity_displacement_field(size, device=None, dtype=None):
    dim = len(size) - 2  # assume NxCxHxW or NxCxDxHxW
    assert dim == 2 or dim == 3
    field_size = list(size)
    field_size[1] = dim
    return torch.zeros(field_size, device=device, dtype=dtype)


def convert_field_from_world_to_torch_space(field, spacing):
    dim = field.shape[1]  # assume displacements with NxCxHxW or NxCxDxHxW
    scale = [2.0/(spacing[d]*field.shape[-(d+1)]) for d in range(dim)]  # multiply with 2/(spacing*size)  (displacements are independent of translations)
    return scale_field(field, scale)


def convert_field_from_torch_to_world_space(field, spacing):
    dim = field.shape[1]  # assume displacements with NxCxHxW or NxCxDxHxW
    scale = [0.5*(spacing[d]*field.shape[-(d+1)]) for d in range(dim)]  # multiply with (spacing*size)/2  (displacements are independent of translations)
    return scale_field(field, scale)


def _convert_to_channel_last(field):
    """Function to permute the input dimensions from the
    DisplacementField convention `(N, 2, H, W)` or `(N, 3, D, H, W)` to the standard PyTorch
    field convention `(N, H, W, 2)` or `(N, D, H, W, 3)`.
    """
    ndims = field.ndimension()
    return field.permute(0, 2, 3, 1) if ndims == 4 else field.permute(0, 2, 3, 4, 1)


def _convert_from_channel_last(field):
    """Function to permute the dimensions of the given field
    from the standard PyTorch field convention `(N, H, W, 2)` to the
    DisplacementField convention `(N, 2, H, W)`.
    """
    ndims = field.ndimension()
    return field.permute(0, 3, 1, 2) if ndims == 4 else field.permute(0, 4, 1, 2, 3)


def scale_field(field, scale):
    assert isinstance(field, torch.Tensor)
    assert field.shape[1] == len(scale)  # assume displacements with NxCxHxW or NxCxDxHxW
    # print('scale with ', scale)
    out_field = field.clone()
    if field.shape[1] == 2:  # 2D displacements
        out_field[:, 0, :, :] *= scale[0]
        out_field[:, 1, :, :] *= scale[1]
    elif field.shape[1] == 3:  # 3D displacements
        out_field[:, 0, :, :, :] *= scale[0]
        out_field[:, 1, :, :, :] *= scale[1]
        out_field[:, 2, :, :, :] *= scale[2]
    else:
        raise TypeError(f'Assume 2D or 3D displacements with NxCxHxW or NxCxDxHxW but got shape {field.shape}.')
    return out_field

#
#  Comparison / check functions
#
def warp_image_sitk(image, field, default_value=0, interpolator=None):
    # fields need to have type sitk.sitkVectorFloat64
    sitk_field = sitk.Cast(field, sitk.sitkVectorFloat64)
    if interpolator is None:
        interpolator = sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.DisplacementFieldTransform(sitk_field))
    out = resampler.Execute(image)
    return out


def warp_image_torch(image, field, grid=None, mode='bilinear', padding_mode='zeros'):
    # DEBUG
    assert isinstance(image, torch.Tensor)
    assert isinstance(field, torch.Tensor)
    if grid is None:
        grid = compute_identity_grid_torch(field.shape, device=field.device, dtype=field.dtype)
    mapping_field = _convert_to_channel_last(grid + field)  # move components to last position
    out = F.grid_sample(image, mapping_field, mode=mode, padding_mode=padding_mode, align_corners=False)
    return out


def get_sitk_interpolator_type(interpolator: str = None):
    interpolator_dict = {'linear': sitk.sitkLinear, 'nearest': sitk.sitkNearestNeighbor, 'cubic': sitk.sitkBSpline,
                         'gaussian': sitk.sitkLabelGaussian}
    return interpolator_dict[interpolator] if interpolator in interpolator_dict else sitk.sitkUnknown


def get_torch_interpolator_type(interpolator: str = None, dim: int = 3):
    interpolator_dict = {'linear': 'bilinear', 'nearest': 'nearest', 'cubic': 'bicubic'} if dim == 2 \
        else {'linear': 'trilinear', 'nearest': 'nearest'}
    return interpolator_dict[interpolator] if interpolator in interpolator_dict else None


if __name__ == '__main__':
    import argparse
    from imi_image_warp import load_image_sitk, load_field_sitk, sitk_image_to_tensor, tensor_image_to_sitk, write_image_sitk
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', required=True, type=str, help='path for input image')
    parser.add_argument('--field', '-f', required=True, type=str, help='path for displacement or velocity field')
    parser.add_argument('--output', '-o', required=True, type=str, help='path for output image')
    parser.add_argument('--linear', '-l', action='store_true', help='Use linear interpolation (default).')
    parser.add_argument('--nearest', '-n', action='store_true', help='Use nearest neighbor interpolation.')
    parser.add_argument('--cubic', '-c', action='store_true', help='Use cubic interpolation.')
    parser.add_argument('--gaussian', '-g', action='store_true', help='Use gaussian label interpolation (use instead of nearest neighbour for label images)')
    parser.add_argument('--torch', action='store_true', help='Use pytorch functions for warping')
    parser.add_argument('--sitk', action='store_true', help='Use SimpleITK functions for warping')
    parser.add_argument('--space', type=str, default='world', help="Space of the field 'world'|'pixel'|'torch', default='world'")
    args = parser.parse_args()

    image = load_image_sitk(args.image)
    field = load_field_sitk(args.field)
    interpolator_type = 'linear' if args.linear else 'nearest' if args.nearest else 'cubic' \
        if args.cubic else 'gaussian' if args.gaussian else 'linear'
    if args.space == 'torch' and args.torch:
        print('warp image (torch space, pytorch) ... ')
        tensor_image, image_meta = sitk_image_to_tensor(image, return_meta_data=True)
        tensor_field, field_meta = sitk_image_to_tensor(field, return_meta_data=True)
        tensor_warp = warp_image_torch(tensor_image, tensor_field, mode=get_torch_interpolator_type(interpolator_type))
        print('Save image ... ')
        write_image_sitk(tensor_image_to_sitk(tensor_warp, image_meta), args.output)
    else:
        tensor_field, field_meta = None, None
        if args.space == 'torch':
            print('convert field (torch space to world space) ... ')
            tensor_field, field_meta = sitk_image_to_tensor(field, return_meta_data=True)
            tensor_field = convert_field_from_torch_to_world_space(tensor_field, field.GetSpacing())
        elif args.space == 'pixel':
            print('convert field (pixel space to world space) ... ')
            tensor_field, field_meta = sitk_image_to_tensor(field, return_meta_data=True)
            tensor_field = convert_field_from_world_to_torch_space(tensor_field, spacing=(1, 1, 1))
            tensor_field = convert_field_from_torch_to_world_space(tensor_field, field.GetSpacing())
        elif args.space == 'world':
            pass
        else:
            raise ValueError(f'Unknown deformation field space of type {args.space}, valid values: world | pixel | torch !')
        warped_image = None
        if args.sitk:
            print('warp image (world space, simpleITK) ... ')
            if tensor_field is not None:
                field = tensor_image_to_sitk(tensor_field, meta_data=field_meta)
            warped_image = warp_image_sitk(image, field, interpolator=get_sitk_interpolator_type(interpolator_type))
        else:
            print('warp image (world space, pytorch) ... ')
            if tensor_field is None:
                tensor_field, field_meta = sitk_image_to_tensor(field, return_meta_data=True)
            tensor_image, image_meta = sitk_image_to_tensor(image, return_meta_data=True)
            warper = ImiDisplacementDeformation(img_size=tensor_image.shape, spacing=image_meta['spacing'])
            tensor_warped = warper(tensor_image,tensor_field)
            warped_image = tensor_image_to_sitk(tensor_warped, image_meta)

        print('Save image ... ')
        write_image_sitk(warped_image, args.output)
    print('Finished.')



