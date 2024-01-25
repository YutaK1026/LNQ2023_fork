import torch
import SimpleITK as sitk
from preregistration.imi_varreg.imi_image_warp import _convert_to_channel_last, _convert_from_channel_last

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


def write_image_sitk(sitk_image, filename: str, pixel_type: str = 'float'):
    assert isinstance(sitk_image, sitk.SimpleITK.Image)

    if sitk_image.GetNumberOfComponentsPerPixel() > 1:
        out_img = sitk_image
    else:
        if pixel_type == 'uint8' and not sitk_image.GetPixelID() == sitk.sitkUInt8:
            out_img = sitk.Clamp(sitk_image, sitk.sitkUInt8)
        elif pixel_type == 'uint16' and not sitk_image.GetPixelID() == sitk.sitkUInt16:
            out_img = sitk.Clamp(sitk_image, sitk.sitkUInt16)
        elif pixel_type == 'int16' and not sitk_image.GetPixelID() == sitk.sitkInt16:
            out_img = sitk.Clamp(sitk_image, sitk.sitkInt16)
        elif pixel_type == 'float' and not sitk_image.GetPixelID() == sitk.sitkFloat32:
            out_img = sitk.Clamp(sitk_image, sitk.sitkFloat32)
        elif pixel_type not in ['uint8', 'uint16', 'int16', 'float']:
            raise ValueError("ERROR in write_image_sitk(): Unsupported pixel type! Supported are 'uint8', "
                             "'uint16', 'int16', 'float'.")
        else:
            out_img = sitk_image

    sitk.WriteImage(out_img, filename)


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
