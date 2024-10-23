import numpy as np
import SimpleITK as sitk
from preregistration.imi_varreg.imi_deformation import scaling_and_squaring
from preregistration.imi_varreg.imi_image_sitk_tools import (
    sitk_image_to_tensor,
    tensor_image_to_sitk,
)


####################################################
#
#  SEGMENTATION HELPERS
#
def label_image_to_reg_mask(label_img):
    reg_labels = (
        [7]
        + [l for l in range(18, 49)]
        + [l for l in range(58, 82)]
        + [l for l in range(84, 88)]
    )
    # print(f"Use reg-labels: {reg_labels}")
    label_array = sitk.GetArrayFromImage(label_img)
    reg_mask = np.zeros(label_array.shape, dtype=label_array.dtype)
    for label in reg_labels:
        reg_mask[label_array == label] = 255
    reg_mask_image = sitk.GetImageFromArray(reg_mask)
    reg_mask_image.CopyInformation(label_img)
    # reg_mask_image = sitk.SmoothingRecursiveGaussian(reg_mask_image, sigma=5.0)
    # print(reg_mask_image)
    return reg_mask_image


def label_image_to_lung_mask(label_img):
    lung_labels = [l for l in range(13, 18)]
    # print(f"Use reg-labels: {reg_labels}")
    label_array = sitk.GetArrayFromImage(label_img)
    lung_mask = np.zeros(label_array.shape, dtype=label_array.dtype)
    for label in lung_labels:
        lung_mask[label_array == label] = 255
    lung_mask_image = sitk.GetImageFromArray(lung_mask)
    lung_mask_image.CopyInformation(label_img)
    # reg_mask_image = sitk.SmoothingRecursiveGaussian(reg_mask_image, sigma=5.0)
    # print(reg_mask_image)
    return lung_mask_image


####################################################
#
#  LINEAR REGISTRATION HELPERS
#
def get_label_center(label_img, label_id):
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(label_img)
    if label_id not in label_stats.GetLabels():
        print(f"Label {label_id} not found!")
        return None
    center = label_stats.GetCentroid(label_id)
    return center


def compute_init_transform(fixed_label, moving_label, label_id):
    fixed_center = get_label_center(fixed_label, label_id)
    moving_center = get_label_center(moving_label, label_id)
    if fixed_center is None or moving_center is None:
        return None
    return [mc - fc for fc, mc in zip(fixed_center, moving_center)]


####################################################
#
#  RESAMPLING HELPERS
#
def resample_image_to_2x2x2(
    sitk_image, interpolator=sitk.sitkLinear, default_value=-1024
):
    old_origin = sitk_image.GetOrigin()
    sitk_image.SetOrigin((0, 0, 0))
    new_spacing = (2, 2, 2)  # (1, 1, 1)
    output_size = [
        int(0.5 + size * os / ns)
        for size, os, ns in zip(
            sitk_image.GetSize(), sitk_image.GetSpacing(), new_spacing
        )
    ]
    sitk_image = sitk.Resample(
        sitk_image,
        interpolator=interpolator,
        size=output_size,
        outputSpacing=new_spacing,
        defaultPixelValue=default_value,
    )
    resampled_origin = sitk_image.GetOrigin()
    final_size = (160, 112, 160)  # (320, 224, 320)  # size comes from the atlas size
    lower_pad = [
        int((nsize - osize) / 2)
        for osize, nsize in zip(sitk_image.GetSize(), final_size)
    ]
    upper_pad = [
        nsize - osize - lpad
        for osize, nsize, lpad in zip(sitk_image.GetSize(), final_size, lower_pad)
    ]
    sitk_image = sitk.ZeroFluxNeumannPad(
        sitk_image, padLowerBound=lower_pad, padUpperBound=upper_pad
    )
    padded_origin = sitk_image.GetOrigin()
    sitk_image.SetOrigin((0, 0, 0))
    print(
        f"  Origins: old={old_origin}, resampled={resampled_origin}, padded={padded_origin}, final={sitk_image.GetOrigin()}"
    )
    return sitk_image


def undo_resample_image_to_2x2x2(
    sitk_image, sitk_reference_image, interpolator=sitk.sitkLinear, default_value=0
):
    old_spacing = sitk_image.GetSpacing()
    old_size = sitk_image.GetSize()
    new_spacing = sitk_reference_image.GetSpacing()
    new_size = sitk_reference_image.GetSize()
    new_origin = sitk_reference_image.GetOrigin()
    #  1. CROPPING
    #  compute original resampled output size
    resampled_output_size = [
        int(0.5 + size * os / ns)
        for size, os, ns in zip(new_size, new_spacing, old_spacing)
    ]
    lower_pad = [
        int((nsize - osize) / 2)
        for osize, nsize in zip(resampled_output_size, old_size)
    ]
    upper_pad = [
        nsize - osize - lpad
        for osize, nsize, lpad in zip(resampled_output_size, old_size, lower_pad)
    ]
    sitk_image.SetOrigin(
        [-(lp * s) for lp, s in zip(lower_pad, old_spacing)]
    )  # WE NEED to do this ... :-/
    cropped_sitk_image = sitk.Crop(
        sitk_image, lowerBoundaryCropSize=lower_pad, upperBoundaryCropSize=upper_pad
    )
    # print(f"Cropped size: {cropped_sitk_image.GetSize()} (lower={lower_pad}, upper={upper_pad}, origin={cropped_sitk_image.GetOrigin()})")
    # 2. Resampling to new size and spacing
    resampled_image = sitk.Resample(
        sitk_image,
        size=new_size,
        outputSpacing=new_spacing,
        interpolator=interpolator,
        defaultPixelValue=default_value,
    )
    resampled_image.SetOrigin(new_origin)
    return resampled_image


####################################################
#
#  VARIATIONAL REGISTRATION HELPERS
#


def get_displacement_from_velocity(sitk_velo_field, inverse=False):
    tensor_field, meta_data = sitk_image_to_tensor(
        sitk_velo_field, return_meta_data=True
    )
    if inverse:
        tensor_displ_field = scaling_and_squaring(
            tensor_field * -1.0, spacing=meta_data["spacing"]
        )
    else:
        tensor_displ_field = scaling_and_squaring(
            tensor_field, spacing=meta_data["spacing"]
        )
    return tensor_image_to_sitk(tensor_displ_field, meta_data=meta_data)


####################################################
#
#  GENERAL HELPERS
#
def compute_dice(atlas_segm, patient_segm):
    atlas_segm = sitk.Cast(atlas_segm, pixelID=sitk.sitkInt16)
    patient_segm = sitk.Cast(patient_segm, pixelID=sitk.sitkInt16)
    label_measures = sitk.LabelOverlapMeasuresImageFilter()
    label_measures.Execute(atlas_segm, patient_segm)
    dice = label_measures.GetDiceCoefficient()
    return dice


def crop_images_to_mask(input_images: list, mask_image: sitk.Image, mask_value=255):
    """Implementation of imiImageExtract in SimpleITK."""
    cropped_output_images = []
    roi_filter = sitk.RegionOfInterestImageFilter()
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(mask_image)
    bounding_box = label_shape_filter.GetBoundingBox(mask_value)
    p1 = mask_image.TransformIndexToPhysicalPoint(
        bounding_box[0 : int(len(bounding_box) / 2)]
    )
    p2 = mask_image.TransformIndexToPhysicalPoint(
        [
            x + sz
            for x, sz in zip(
                bounding_box[0 : int(len(bounding_box) / 2)],
                bounding_box[int(len(bounding_box) / 2) :],
            )
        ]
    )
    for input_image in input_images:
        start_index = input_image.TransformPhysicalPointToIndex(p1)
        end_index = input_image.TransformPhysicalPointToIndex(p2)
        roi_filter.SetSize(
            (
                end_index[0] - start_index[0],
                end_index[1] - start_index[1],
                end_index[2] - start_index[2],
            )
        )
        roi_filter.SetIndex(start_index)
        result = roi_filter.Execute(input_image)
        cropped_output_images.append(result)
    return cropped_output_images


def extract_slices(sitk_image, axis="y", is_ct=True):
    image_size = list(sitk_image.GetSize())
    if axis == "y":
        out_size = [image_size[0], 0, image_size[2]]
        out_index = [0, image_size[1] // 2, 0]
    elif axis == "x":
        out_size = [0, image_size[1], image_size[2]]
        out_index = [image_size[0] // 2, 0, 0]
    elif axis == "z":
        out_size = [image_size[0], image_size[1], 0]
        out_index = [0, 0, image_size[2] // 2]
    else:
        raise ValueError(f"Unknown axis {axis}.")
    sitk_slice = sitk.Extract(sitk_image, size=out_size, index=out_index)
    if is_ct:
        sitk_slice = sitk.Cast(
            sitk.IntensityWindowing(
                sitk_slice,
                windowMinimum=-200,
                windowMaximum=300,
                outputMinimum=0,
                outputMaximum=255,
            ),
            pixelID=sitk.sitkUInt8,
        )
    else:
        sitk_slice = sitk.Cast(
            sitk.RescaleIntensity(sitk_slice, outputMinimum=0, outputMaximum=255),
            pixelID=sitk.sitkUInt8,
        )

    sitk_slice = sitk.Flip(sitk_slice, flipAxes=(False, True))
    return sitk_slice
