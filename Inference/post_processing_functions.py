import os
from pathlib import Path

import numpy as np
import scipy
import SimpleITK as sitk
import torch


def flood_fill_hull(image: np.ndarray) -> tuple:
    """Source: https://stackoverflow.com/questions/46310603/how-to-compute-convex-hull-image-volume-in-3d-numpy-arrays"""
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


def get_probability_threshold_constant_thres(predictions_shape: str, threshold: float):
    probability_threshold = np.full(shape=predictions_shape, fill_value=threshold)
    return probability_threshold


def get_probability_threshold_probmap(map: str, threshold: float):
    map = sitk.GetArrayFromImage(sitk.ReadImage(map))
    map_norm = 0.5 * np.round((map - map.min()) / (map.max() - map.min()), 2)
    probability_threshold = np.full(shape=map.shape, fill_value=threshold)
    probability_threshold = (1 - map_norm) * probability_threshold
    return probability_threshold


def get_final_predictions(predictions: np.ndarray, probability_threshold: np.ndarray):
    final_predictions = np.zeros(predictions.shape).astype("uint8")
    final_predictions[predictions < probability_threshold] = 0
    final_predictions[predictions >= probability_threshold] = 1
    final_predictions_itk = sitk.GetImageFromArray(final_predictions)
    return final_predictions_itk


def cut_predictions(predictions: np.ndarray, probm_path: str, threshold: float):
    probability_threshold = get_probability_threshold_probmap(probm_path, threshold)
    final_predictions_itk = get_final_predictions(predictions, probability_threshold)
    return final_predictions_itk


def create_regmask(lungmask) -> str:
    """Create convex hull of the lung."""
    lung = sitk.ReadImage(lungmask)
    regmask_arr = sitk.GetArrayFromImage(lung)
    out_img, hull = flood_fill_hull(regmask_arr)
    img = sitk.GetImageFromArray(out_img)
    img.SetSpacing(lung.GetSpacing())
    img.SetOrigin(lung.GetOrigin())
    img = sitk.Cast(img, sitk.sitkUInt8)
    patient_id = Path(lungmask).name.split("_lung_mask")[0]
    sitk.WriteImage(
        img, os.path.join(Path(lungmask).parent, f"{patient_id}_regmask.nii.gz")
    )
    print(f"{patient_id} is done.")
    return os.path.join(Path(lungmask).parent, f"{patient_id}_regmask.nii.gz")


def remove_small_regions(img):
    """All connected components smaller than size = 5 are removed."""
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    mask_relabeled = connected_component_filter.Execute(img)
    # find the largest/two largest objects
    filter_rel = sitk.RelabelComponentImageFilter()
    filter_rel.Execute(mask_relabeled)
    object_sizes = filter_rel.GetSizeOfObjectsInPixels()
    for idx, size in enumerate(object_sizes):
        if size < 5:
            filter_rel.SetMinimumObjectSize(object_sizes[idx - 1])
            break
    img = filter_rel.Execute(mask_relabeled)
    img = sitk.Clamp(img, upperBound=1)
    return img


def pad_to_original_size(img_new_size, img_orig_size):
    """Pad labels to original image size."""
    img = sitk.Resample(img_new_size, img_orig_size, defaultPixelValue=0)
    return img


def remove_outside_lung(mask_img, regmask_img):
    """Self-explanatory."""
    assert mask_img.GetSize() == regmask_img.GetSize()
    return sitk.Mask(mask_img, regmask_img)


def remove_healthy_lymphnodes_sitk(mask_img, diameter: int):
    """Remove segmentations that do not fulfill RECIST."""
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    mask_relabeled = connected_component_filter.Execute(mask_img)
    filter_rel = sitk.RelabelComponentImageFilter()
    filter_rel.Execute(mask_relabeled)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_relabeled)
    relabelmap = {}
    for i in stats.GetLabels():
        keep_component = True
        for dia in stats.GetEquivalentEllipsoidDiameter(i):
            if dia < diameter:
                keep_component = False
                break
        if keep_component:
            relabelmap[i] = 1
        else:
            relabelmap[i] = 0
    output = sitk.ChangeLabel(mask_relabeled, changeMap=relabelmap)
    return output
