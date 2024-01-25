import torch
import torch.nn.functional as F

eps = 1e-6


def normalize_image(image, normalization_type="scale01"):
    if normalization_type == "scale01":
        imin, imax = image.min(), image.max()
        out_image = (image - imin) / (imax - imin) if (imax - imin) > eps else image - imin
        denorm_info = {'min': imin, 'max': imax}
    else:
        raise ValueError(f"Unknown normalization type '{normalization_type}'! Supported types are 'scale01'.")
    return out_image, denorm_info


def denormalize_image(image, denormalize_info, normalization_type="scale01"):
    if normalization_type == "scale01":
        imin, imax = denormalize_info['min'], denormalize_info['max']
        out_image = image * (imax - imin) + imin if (imax - imin) > eps else image + imin
    else:
        raise ValueError(f"Unknown normalization type '{normalization_type}'! Supported types are 'scale01'.")
    return out_image


def crop_image(image, new_size, start=(0, 0, 0)):
    dim = len(image.size()) - 2
    assert dim == 2 or dim == 3
    assert len(new_size) == dim and len(start) == dim
    return image[:, :, start[0]:new_size[0], start[1]:new_size[1], start[2]:new_size[2]] \
        if dim == 3 else image[:, :, start[0]:new_size[0], start[1]:new_size[1]]


def pad_image_to_size(image, new_size, pad_mode='end', fill_mode='constant', fill_value=0.0):
    dim = len(image.size()) - 2
    assert dim == 2 or dim == 3
    assert len(new_size) == dim
    input_size = image.shape[2:]
    padding_vals = [new_size[-(d + 1)] - input_size[-(d + 1)] for d in range(dim)]
    if pad_mode == 'end':
        padding = torch.tensor(list(zip([0]*dim, padding_vals))).flatten()
    elif pad_mode == 'start':
        padding = torch.tensor(list(zip(padding_vals, [0]*dim))).flatten()
    return F.pad(image, list(padding), mode=fill_mode, value=fill_value)


def compute_image_statistics(image, name=""):
    statistics = {'name': name, 'mean': image.mean(), 'min': image.min(), 'max': image.max(),
                  'sqr_mean': image.pow(2).mean(), '#zeros': torch.count_nonzero(image.abs() <= 1e-8)}
    return statistics


def compute_field_statistics(field, name="", per_component=False):
    assert len(field.size()) == 4 or len(field.size()) == 5
    field_magnitude_sqr = field.pow(2).sum(dim=1)
    field_magnitude = field_magnitude_sqr.sqrt()
    statistics = {'name': name, 'mean': field_magnitude.mean(), 'min': field_magnitude.min(), 'max': field_magnitude.max(),
                  'sqr_mean': field_magnitude_sqr.mean(), '#zeros': torch.count_nonzero(field_magnitude <= 1e-10)}
    if per_component:
        for c in range(field.shape[1]):
            component_stat = compute_image_statistics(field[:, c, ...])
            for key, value in component_stat.items():
                if key != 'name':
                    key = f"component {c} {key}"
                    statistics[key] = value
    return statistics


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
