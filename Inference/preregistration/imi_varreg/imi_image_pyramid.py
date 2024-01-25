import torch
import torch.nn.functional as F
from typing import List
from preregistration.imi_varreg.gaussian_smoothing import GaussianSmoothing


def pyrdown(input: torch.Tensor, border_type: str = 'reflect',
            align_corners: bool = False, factor: float = 2.0) -> torch.Tensor:
    r"""Blur a tensor and downsamples it.

    .. image:: _static/img/pyrdown.png

    Args:
        input: the tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    dim = len(input.shape) - 2
    if not (len(input.shape) == 4 or len(input.shape) == 5):
        raise ValueError(f"Invalid input shape, we expect NxCxHxW or NxCxDxHxW. Got: {input.shape}")
    sigma = 0.5 * factor
    kernel_size = 2 * round((2.5*sigma)) + 1
    smoother = GaussianSmoothing(channels=input.shape[1], kernel_size=kernel_size, sigma=sigma, dim=dim).to(device=input.device)
    # blur image
    x_blur: torch.Tensor = smoother(input)
    # downsample.
    if len(input.shape) == 4:  # 2D images
        _, _, height, width = input.shape
        out: torch.Tensor = F.interpolate(
            x_blur,
            size=(int(float(height) / factor), int(float(width) / factor)),
            mode='bilinear',
            align_corners=align_corners
        )
    else:  # 3D images
        _, _, depth, height, width = input.shape
        print(input.shape, (int(float(depth) / factor), int(float(height) / factor), int(float(width) / factor)))
        out: torch.Tensor = F.interpolate(
            x_blur,
            size=(int(float(depth) / factor), int(float(height) / factor), int(float(width) / factor)),
            mode='trilinear',
            align_corners=align_corners
        )
    return out


def pyrup(input: torch.Tensor, border_type: str = 'reflect', align_corners: bool = False) -> torch.Tensor:
    r"""Upsample a tensor and then blurs it.

    .. image:: _static/img/pyrup.png

    Args:
        input: the tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the downsampled tensor.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> pyrup(input, align_corners=True)
        tensor([[[[0.7500, 0.8750, 1.1250, 1.2500],
                  [1.0000, 1.1250, 1.3750, 1.5000],
                  [1.5000, 1.6250, 1.8750, 2.0000],
                  [1.7500, 1.8750, 2.1250, 2.2500]]]])
    """
    if not (len(input.shape) == 4 or len(input.shape) == 5):
        raise ValueError(f"Invalid input shape, we expect NxCxHxW or NxCxDxHxW. Got: {input.shape}")
    smoother = GaussianSmoothing(channels=input.shape[1], kernel_size=5, sigma=1, dim=len(input.shape)-2).to(device=input.device)
    # upsample tensor
    if len(input.shape) == 4:  # 2D images
        _, _, height, width = input.shape
        x_up: torch.Tensor = F.interpolate(
            input, size=(height * 2, width * 2), mode='bilinear', align_corners=align_corners
        )
    else:
        _, _, depth, height, width = input.shape
        x_up: torch.Tensor = F.interpolate(
            input, size=(depth * 2, height * 2, width * 2), mode='trilinear', align_corners=align_corners
        )
    # blurs upsampled tensor
    # x_blur: torch.Tensor = smoother(x_up)
    return x_up  # x_blur


def build_pyramid(
    input: torch.Tensor, max_level: int, border_type: str = 'reflect', align_corners: bool = False
) -> List[torch.Tensor]:
    r"""Construct the Gaussian pyramid for an image.

    .. image:: _static/img/build_pyramid.png

    The function constructs a vector of images and builds the Gaussian pyramid
    by recursively applying pyrDown to the previously built pyramid layers.

    Args:
        input : the tensor to be used to construct the pyramid.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not (len(input.shape) == 4 or len(input.shape) == 5):
        raise ValueError(f"Invalid input shape, we expect NxCxHxW or NxCxDxHxW. Got: {input.shape}")

    if not isinstance(max_level, int) or max_level < 0:
        raise ValueError(f"Invalid max_level, it must be a positive integer. Got: {max_level}")

    # create empty list and append the original image
    #pyramid: List[torch.Tensor] = [input]

    # iterate and downsample
    #for _ in range(max_level - 1):
    #    img_curr: torch.Tensor = pyramid[-1]
    #    img_down: torch.Tensor = pyrdown(img_curr, border_type, align_corners)
    #    pyramid.append(img_down)

    # create empty list and append the original image
    pyramid: List[torch.Tensor] = []

    # iterate and downsample
    for l in range(max_level):
        img_down: torch.Tensor = pyrdown(input, border_type, align_corners, factor=2 ** l)
        pyramid.append(img_down)
    return pyramid
