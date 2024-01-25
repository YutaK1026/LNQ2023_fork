import argparse
import torch
import torch.nn.functional as F
import SimpleITK as sitk

@torch.no_grad()
def compute_identity_grid(size, device=None, dtype=None):
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


def get_meta_data_dict(sitk_image):
    assert isinstance(sitk_image, sitk.SimpleITK.Image)
    meta_data_dict = {}
    meta_data_dict['filename'] = ""
    meta_data_dict['dim'] = sitk_image.GetDimension()
    meta_data_dict['size'] = sitk_image.GetSize()
    meta_data_dict['spacing'] = sitk_image.GetSpacing()
    meta_data_dict['origin'] = sitk_image.GetOrigin()
    meta_data_dict['direction'] = sitk_image.GetDirection()
    # Todo: add further informations ...
    return meta_data_dict


def set_meta_data(sitk_image, meta_data: dict):
    assert isinstance(sitk_image, sitk.SimpleITK.Image)
    assert sitk_image.GetDimension() == meta_data['dim']
    if meta_data['size'] != sitk_image.GetSize():
        print(f"WARNING: image and meta-data has different size {sitk_image.GetSize()} <> {meta_data['size']} !")
    sitk_image.SetSpacing(meta_data['spacing'])
    sitk_image.SetOrigin(meta_data['origin'])
    sitk_image.SetDirection(meta_data['direction'])


def sitk_image_to_tensor(image, return_meta_data=False):
    if isinstance(image, torch.Tensor):
        return image if not return_meta_data else (image, None)
    elif isinstance(image, sitk.SimpleITK.Image):
        tensor_image = torch.tensor(sitk.GetArrayFromImage(image).squeeze())
        # convert to
        while tensor_image.dim() < image.GetDimension() + 2:
            tensor_image = tensor_image.unsqueeze(0)
        # DEBUG
        assert tensor_image.dim() == image.GetDimension() + 2
        if image.GetNumberOfComponentsPerPixel() > 1:
            tensor_image = _convert_from_channel_last(tensor_image)  # default from sitk is HxWxC -> to NxCxHxW
        return tensor_image if not return_meta_data else (tensor_image, get_meta_data_dict(image))
    else:
        raise TypeError('ERROR in sitk_image_to_tensor(): Only torch tensor or SimpleITK Image '
                        'types are supported!')


def tensor_image_to_sitk(image, meta_data=None):
    assert isinstance(image, torch.Tensor)
    assert image.dim() == 4 or image.dim() == 5, f'Assume tensor images in NxCxHxW or NxCxDxHxW but got {image.shape}.'
    if image.shape[0] > 1:
        raise ValueError(f'Conversion of batches of torch images yet not implemented!')
    is_vector = image.shape[1] > 1  # multiple channels
    if is_vector:
        image = _convert_to_channel_last(image)  # sitk needs channel last convention
    sitk_image = sitk.GetImageFromArray(image.squeeze().cpu().numpy(), isVector=is_vector)
    if meta_data is not None:
        set_meta_data(sitk_image, meta_data)
    return sitk_image


def load_field_sitk(filename: str):
    sitk_image = load_image_sitk(filename, pixel_type=sitk.sitkUnknown)
    if sitk_image.GetNumberOfComponentsPerPixel() < 2:
        raise TypeError(f'Expected field image (multiple components) in {filename}!')
    return sitk_image


def load_image_sitk(filename: str, pixel_type=sitk.sitkFloat32):
    sitk_image = sitk.ReadImage(filename, outputPixelType=pixel_type)
    return sitk_image


def get_sitk_interpolator_type(interpolator: str = None):
    interpolator_dict = {'linear': sitk.sitkLinear, 'nearest': sitk.sitkNearestNeighbor, 'cubic': sitk.sitkBSpline,
                         'gaussian': sitk.sitkLabelGaussian}
    return interpolator_dict[interpolator] if interpolator in interpolator_dict else sitk.sitkUnknown


def get_torch_interpolator_type(interpolator: str = None, dim: int = 3):
    interpolator_dict = {'linear': 'bilinear', 'nearest': 'nearest', 'cubic': 'bicubic'} if dim == 2 \
        else {'linear': 'trilinear', 'nearest': 'nearest'}
    return interpolator_dict[interpolator] if interpolator in interpolator_dict else None


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


def warp_image_with_torch(image, field, grid=None, mode='bilinear', padding_mode='zeros'):
    # DEBUG
    assert isinstance(image, torch.Tensor)
    assert isinstance(field, torch.Tensor)
    if grid is None:
        grid = compute_identity_grid(field.shape, device=field.device, dtype=field.dtype)
    field = field + grid
    mapping_field = _convert_to_channel_last(field)  # move components to last position
    # out = image_sample(input_image, mapping_field, mode=mode, padding_mode=padding_mode, align_corners=False)
    out = F.grid_sample(image, mapping_field, mode=mode, padding_mode=padding_mode, align_corners=False)
    return out


def write_image_sitk(sitk_image, filename: str, pixel_type: str = 'float'):
    assert isinstance(sitk_image, sitk.SimpleITK.Image)

    if sitk_image.GetNumberOfComponentsPerPixel() > 1:
        out_img = sitk_image
    else:
        if pixel_type is None:
            out_img = sitk_image
        elif pixel_type == 'uint8' and not sitk_image.GetPixelID() == sitk.sitkUInt8:
            out_img = sitk.Clamp(sitk_image, sitk.sitkUInt8)
        elif pixel_type == 'uint16' and not sitk_image.GetPixelID() == sitk.sitkUInt16:
            out_img = sitk.Clamp(sitk_image, sitk.sitkUInt16)
        elif pixel_type == 'int16' and not sitk_image.GetPixelID() == sitk.sitkInt16:
            out_img = sitk.Clamp(sitk_image, sitk.sitkInt16)
        elif pixel_type == 'float' and not sitk_image.GetPixelID() == sitk.sitkFloat32:
            out_img = sitk.Clamp(sitk_image, sitk.sitkFloat32)
        elif pixel_type is not None and pixel_type not in ['uint8', 'uint16', 'int16', 'float']:
            raise ValueError("ERROR in write_image_sitk(): Unsupported pixel type! Supported are 'uint8', "
                             "'uint16', 'int16', 'float'.")
        else:
            out_img = sitk_image

    sitk.WriteImage(out_img, filename)


def convert_field_from_pixel_to_torch_space(field):
    dim = field.shape[1]  # assume displacements with NxCxHxW or NxCxDxHxW
    scale = [2.0/field.shape[-(d+1)] for d in range(dim)]  # multiply with 2/size  (displacements are independent of translations)
    shift = [0] * len(scale)
    return scale_shift_field(field, scale, shift)


def scale_shift_field(field, scale, shift):
    assert isinstance(field, torch.Tensor)
    assert field.shape[1] == len(scale)  # assume displacements with NxCxHxW or NxCxDxHxW
    assert field.shape[1] == len(shift)
    print('scale shift with ', scale, shift)
    out_field = field.clone()
    if field.shape[1] == 2:  # 2D displacements
        out_field[:, 0, :, :] *= scale[0] + shift[0]
        out_field[:, 1, :, :] *= scale[1] + shift[1]
    elif field.shape[1] == 3:  # 3D displacements
        out_field[:, 0, :, :, :] *= scale[0] + shift[0]
        out_field[:, 1, :, :, :] *= scale[1] + shift[1]
        out_field[:, 2, :, :, :] *= scale[2] + shift[2]
    else:
        raise TypeError(f'Assume 2D or 3D displacements with NxCxHxW or NxCxDxHxW but got shape {field.shape}.')
    return out_field


if __name__ == '__main__':
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', required=True, type=str, help='path for input image')
    parser.add_argument('--field', '-f', required=True, type=str, help='path for displacement or velocity field')
    parser.add_argument('--output', '-o', required=True, type=str, help='path for output image')
    parser.add_argument('--linear', '-l', action='store_true', help='Use linear interpolation (default).')
    parser.add_argument('--nearest', '-n', action='store_true', help='Use nearest neighbor interpolation.')
    parser.add_argument('--cubic', '-c', action='store_true', help='Use cubic interpolation.')
    parser.add_argument('--gaussian', '-g', action='store_true', help='Use gaussian label interpolation (use instead of nearest neighbour for label images)')
    args = parser.parse_args()

    image = load_image_sitk(args.image)
    field = load_field_sitk(args.field)
    interpolator_type = 'linear' if args.linear else 'nearest' if args.nearest else 'cubic' \
        if args.cubic else 'gaussian' if args.gaussian else 'linear'
    sitk_interp_type = get_sitk_interpolator_type(interpolator_type)
    warp_image = warp_image_sitk(image, field, interpolator=sitk_interp_type)
    write_image_sitk(warp_image, args.output)

    tensor_image, image_meta = sitk_image_to_tensor(image, return_meta_data=True)
    tensor_field = sitk_image_to_tensor(field)
    grid = compute_identity_grid(tensor_image.shape)
    print(tensor_image.shape, tensor_field.shape, grid.shape)
    print(field.GetPixel(20, 30, 40))
    print(tensor_field[0,:,40,30,20])
    #tensor_field = tensor_field[:, [2, 1, 0], :, :, :]
    #print(tensor_field[0,:,40,30,20])

    tensor_field = convert_field_from_pixel_to_torch_space(convert_field_from_world_to_pixel_space(tensor_field, field.GetSpacing()))
    warp2 = warp_image_with_torch(tensor_image, tensor_field, grid=grid)

    write_image_sitk(tensor_image_to_sitk(warp2, image_meta), args.output)



