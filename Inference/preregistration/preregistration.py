import os
import glob
import time
import numpy as np
import SimpleITK as sitk
import subprocess
from preregistration.data import PatientCase, Atlas, ensure_path
from preregistration.sitk_linear_registration import (sitk_register_similarity, sitk_register_affine, sitk_inverse_linear_transform,
                                      sitk_transform_ct_image, sitk_transform_mask_image)
from preregistration.imi_varreg.imi_variational_registration import run_var_reg
from preregistration.imi_varreg.imi_image_sitk_tools import warp_image_sitk
from preregistration.prereg_utils import (label_image_to_reg_mask, label_image_to_lung_mask, compute_init_transform,
                          resample_image_to_2x2x2, undo_resample_image_to_2x2x2, get_displacement_from_velocity,
                          compute_dice, crop_images_to_mask, extract_slices)

save_images=False


def total_segmentation(patient: PatientCase, force=False, debug=True):
    print(f"RUN TotalSegmentator for patient id {patient.patient_id}")
    if not patient.is_available('total_segm') or force:
        ct_filename = patient.filenames['ct']
        print(ct_filename)
        output_segm_filename = patient.filenames['total_segm']
        print(f"  make segmentation and write to {output_segm_filename}")
        try:
            cmd = f'source /opt/app/venv/bin/activate; TotalSegmentator -i {ct_filename} -o {output_segm_filename} -ml -f; deactivate'
            subprocess.run(cmd, shell=True, executable='/bin/bash', cwd='/home/user', check=True)
        except subprocess.CalledProcessError as e:
            cmd = f'source /opt/app/venv/bin/activate; TotalSegmentator -i {ct_filename} -o {output_segm_filename} -ml -f -fs -bs; deactivate'
            subprocess.run(cmd, shell=True, executable='/bin/bash', cwd='/home/user')
        #np_segm_image = totalsegmentator(ct_filename, output_segm_filename, ml=True, verbose=False)
    if not patient.is_available('reg_mask') or force:
        print(f"  make reg mask and write to {patient.filenames['reg_mask']}")
        label_img = patient.get_image('total_segm')
        reg_mask_image = label_image_to_reg_mask(label_img)
        patient.set_image('reg_mask', reg_mask_image, save_image=save_images)
        if debug:
            out_png_dir = os.path.join(patient.png_dir, "reg_masks")
            ensure_path(out_png_dir)
            reg_mask_slice = extract_slices(reg_mask_image, axis='x', is_ct=False)
            sitk.WriteImage(reg_mask_slice, os.path.join(out_png_dir, f"{patient.patient_id}_reg_mask.png"))
    if not patient.is_available('lung_mask') or force:
        print(f"  make lung mask and write to {patient.filenames['lung_mask']}")
        label_img = patient.get_image('total_segm')
        lung_mask_image = label_image_to_lung_mask(label_img)
        patient.set_image('lung_mask', lung_mask_image, save_image=save_images)
        if debug:
            out_png_dir = os.path.join(patient.png_dir, "lung_masks")
            ensure_path(out_png_dir)
            reg_mask_slice = extract_slices(lung_mask_image, axis='x', is_ct=False)
            sitk.WriteImage(reg_mask_slice, os.path.join(out_png_dir, f"{patient.patient_id}_lung_mask.png"))


def smooth_mask(sitk_mask):
    return sitk.SmoothingRecursiveGaussian(sitk_mask, sigma=2.0)


def linear_registration(patient: PatientCase, atlas: Atlas, force=False, debug=True):
    print(f"RUN linear registration for patient id {patient.patient_id}")
    if patient.is_available('transform_affine') and patient.is_available('transform_rigid') and not force:
        print(f"  SKIP linear registration because already done!")
        return
    atlas_reg_mask = smooth_mask(atlas.get_image('reg_mask'))  # moving
    pat_reg_mask = smooth_mask(patient.get_image('reg_mask'))  # fixed
    # RIGID REGISTRATION
    if not patient.is_available('transform_rigid') or force:
        print(f"  RIGID registration for patient id {patient.patient_id}.")
        atlas_segm = atlas.get_image('total_segm')
        pat_segm_img = patient.get_image('total_segm')
        init_translation = compute_init_transform(pat_segm_img, atlas_segm, label_id=46)
        rigid_transform, moving_resampled = sitk_register_similarity(fixed_image=pat_reg_mask,
                                                                     moving_image=atlas_reg_mask,
                                                                     init_translation=init_translation)
        patient.set_image('transform_rigid', rigid_transform, save_image=save_images)
    # ALWAYS perform AFFINE REGISTRATION
    print(f"  AFFINE registration for patient id {patient.patient_id}.")
    affine_transform, moving_resampled = sitk_register_affine(fixed_image=pat_reg_mask,
                                                              moving_image=atlas_reg_mask,
                                                              initial_transform=patient.get_image('transform_rigid'))
    patient.set_image('transform_affine', affine_transform, save_image=save_images)
    if debug:  # not really needed
        # WARP patient CT to atlas
        inverse_affine_transform = sitk_inverse_linear_transform(affine_transform)
        pat_ct_affine = sitk_transform_ct_image(transform=inverse_affine_transform, fixed_image=atlas_reg_mask,
                                                moving_image=patient.get_image('ct'))
        patient.set_image('ct_affine', pat_ct_affine, save_image=save_images)
        pat_reg_mask_affine = sitk_transform_mask_image(transform=inverse_affine_transform, fixed_image=atlas_reg_mask,
                                                        moving_image=pat_reg_mask)
        patient.set_image('mask_affine', pat_reg_mask_affine, save_image=save_images)
    if debug:
        out_png_dir = os.path.join(patient.png_dir, "ct_affine")
        ensure_path(out_png_dir)
        pat_ct_affine_slice = extract_slices(pat_ct_affine, axis='y')
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"{patient.patient_id}_ct_affine.png"))


def resample_images(patient: PatientCase, atlas: Atlas, force=False, debug=True):
    print(f"RUN resampling for patient id {patient.patient_id}")
    if patient.is_available('ct_affine_resample') and patient.is_available('segm_affine_resample') and not force:
        print(f"  SKIP resampling because already done!")
        return
    atlas_reg_mask = atlas.get_image('reg_mask')  # fixed reference for transforms
    pat_ct_image = patient.get_image('ct')
    pat_segm_image = patient.get_image('total_segm')
    pat_to_atlas_transform = sitk_inverse_linear_transform(patient.get_image('transform_affine'))
    print("  transform ct and segm to atlas")
    pat_to_atlas_ct_image = sitk_transform_ct_image(pat_to_atlas_transform, fixed_image=atlas_reg_mask,
                                                    moving_image=pat_ct_image)
    pat_to_atlas_segm_image = sitk_transform_mask_image(pat_to_atlas_transform, fixed_image=atlas_reg_mask,
                                                        moving_image=pat_segm_image)
    print("  resample ct and segm to resolution 2x2x2")
    pat_to_atlas_ct_image_resampled = resample_image_to_2x2x2(pat_to_atlas_ct_image, interpolator=sitk.sitkLinear,
                                                              default_value=-1024)
    pat_to_atlas_segm_image_resampled = resample_image_to_2x2x2(pat_to_atlas_segm_image,
                                                                interpolator=sitk.sitkLabelGaussian, default_value=0)
    patient.set_image('ct_affine_resample', pat_to_atlas_ct_image_resampled, save_image=save_images)
    patient.set_image('segm_affine_resample', pat_to_atlas_segm_image_resampled, save_image=save_images)
    if debug:
        out_png_dir = os.path.join(patient.png_dir, "ct_resample")
        ensure_path(out_png_dir)
        pat_ct_affine_slice = extract_slices(patient.get_image('ct_affine_resample'), axis='y')
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"{patient.patient_id}_ct_affine_resample.png"))
        pat_ct_affine_slice = extract_slices(patient.get_image('ct_affine_resample'), axis='x')
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"sag_{patient.patient_id}_ct_affine_resample.png"))
    if debug:
        out_png_dir = os.path.join(patient.png_dir, "segm_resample")
        ensure_path(out_png_dir)
        pat_ct_affine_slice = extract_slices(patient.get_image('segm_affine_resample'), axis='y', is_ct=False)
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"{patient.patient_id}_segm_affine_resample.png"))
        pat_ct_affine_slice = extract_slices(patient.get_image('segm_affine_resample'), axis='x', is_ct=False)
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"sag_{patient.patient_id}_segm_affine_resample.png"))


def variational_registration(patient: PatientCase, atlas: Atlas, device='cuda', force=False, debug=True):
    print(f"RUN variational registration for patient id {patient.patient_id}")
    if patient.is_available('to_atlas_displ') and patient.is_available('to_atlas_velo') and not force:
        print(f"  SKIP variational registration because already done!")
        return
    atlas_segm_image = atlas.get_image('segm_resample', as_pixel_type=sitk.sitkFloat32)  # fixed image
    pat_segm_affine_image = patient.get_image('segm_affine_resample', as_pixel_type=sitk.sitkFloat32)  # moving image
    print("  run variational registration")
    output_varreg_dict = run_var_reg(sitk_fixed_image=atlas_segm_image, sitk_moving_image=pat_segm_affine_image,
                                     iterations=800, alpha=0.5, device=device)
    patient.set_image('to_atlas_displ', output_varreg_dict['displacement'], save_image=save_images)
    patient.set_image('to_atlas_velo', output_varreg_dict['velocity'], save_image=save_images)
    print("  warp ct and segm images")
    displ_field = output_varreg_dict['displacement']
    pat_segm_warped_image = warp_image_sitk(pat_segm_affine_image, displ_field, default_value=0, interpolator=sitk.sitkNearestNeighbor)
    patient.set_image('to_atlas_segm_warped', pat_segm_warped_image, save_image=save_images)
    output_varreg_dict['dice_before'] = compute_dice(atlas_segm_image, pat_segm_affine_image)
    output_varreg_dict['dice_after'] = compute_dice(atlas_segm_image, pat_segm_warped_image)
    output_varreg_dict['patid'] = patient.patient_id
    print(f"DICE scores of patient id {patient.patient_id}: before={output_varreg_dict['dice_before']} after={output_varreg_dict['dice_after']}")
    if debug:
        pat_ct_affine_image = patient.get_image('ct_affine_resample')
        pat_ct_warped_image = warp_image_sitk(pat_ct_affine_image, displ_field, default_value=-1024,
                                              interpolator=sitk.sitkLinear)
        patient.set_image('to_atlas_ct_warped', pat_ct_warped_image, save_image=True)
    if debug:
        out_csv_file = patient.filenames['to_atlas_displ'].replace('to_atlas_displ.nii.gz', 'to_atlas_stats.csv')
        print(f"Save output to csv {out_csv_file}.")
        with open(out_csv_file, 'w') as f:
            stats_keys = ['patid', 'best_loss', 'dice_before', 'dice_after', 'mse', 'mae', 'reg_diff',
                          'log_jacobian', 'std_jacobian', 'neg_jacobian']
            f.write("# ")
            for header in stats_keys:
                f.write(f"{header};  ")
            f.write("\n")
            for key in stats_keys:
                f.write(str(output_varreg_dict[key]) + "; ")
            f.write("\n")
        out_png_dir = os.path.join(patient.png_dir, "var_reg")
        ensure_path(out_png_dir)
        pat_ct_affine_slice = extract_slices(patient.get_image('to_atlas_ct_warped'), axis='y', is_ct=True)
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"{patient.patient_id}_to_atlas_ct_warped.png"))
        pat_ct_affine_slice = extract_slices(patient.get_image('to_atlas_ct_warped'), axis='x', is_ct=True)
        sitk.WriteImage(pat_ct_affine_slice, os.path.join(out_png_dir, f"sag_{patient.patient_id}_to_atlas_ct_warped.png"))


def transform_atlas_maps_to_patient(patient: PatientCase, atlas: Atlas, force=False, debug=True):
    print(f"RUN transform atlas maps to patient id {patient.patient_id}")
    if patient.is_available('to_atlas_displ') and patient.is_available('to_atlas_velo') and not force:
        print(f"  SKIP variational registration because already done!")
        return
    print("  build composite transform from inverse warping and affine registration")
    atlas_to_pat_affine_transform = patient.get_image('transform_affine')
    pat_to_atlas_velocity = patient.get_image('to_atlas_velo')
    atlas_to_pat_displ = get_displacement_from_velocity(pat_to_atlas_velocity, inverse=True)
    # print(f"{atlas_to_pat_displ.GetSize()} {atlas_to_pat_displ.GetSpacing()} {atlas_to_pat_displ.GetOrigin()}")
    orig_size_atlas_image = atlas.get_image('total_segm')  # faster to load (and needed anyway) as atlas.get_image('ct')
    atlas_to_pat_displ = undo_resample_image_to_2x2x2(atlas_to_pat_displ, sitk_reference_image=orig_size_atlas_image,
                                                      interpolator=sitk.sitkLinear, default_value=0)
    atlas_to_pat_displ = sitk.Cast(atlas_to_pat_displ, sitk.sitkVectorFloat64)
    # print(f"{atlas_to_pat_displ.GetSize()} {atlas_to_pat_displ.GetSpacing()} {atlas_to_pat_displ.GetOrigin()}")
    # one transform for all steps
    composite_transform = sitk.CompositeTransform(3)
    composite_transform.AddTransform(sitk.DisplacementFieldTransform(atlas_to_pat_displ))
    composite_transform.AddTransform(atlas_to_pat_affine_transform)
    print("  transform atlas distance map to patient")
    patient_ct_img = patient.get_image('ct')
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(patient_ct_img)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(atlas.coord_map_bg_value)
    resampler.SetTransform(composite_transform)
    patient_distance_map = sitk.VectorMagnitude(resampler.Execute(atlas.atlas_coordinate_map))
    patient_distance_map = sitk.Cast(patient_distance_map / 273.7, sitk.sitkFloat32)
    patient.set_image('distance_map', patient_distance_map, save_image=True)
    print("  transform atlas label probs map to patient")
    avg_label_map = atlas.get_image('average_labels')
    resampler.SetDefaultPixelValue(0.0)
    atlas_to_pat_avg_label_map = resampler.Execute(avg_label_map)
    atlas_to_pat_avg_label_map = sitk.RecursiveGaussian(atlas_to_pat_avg_label_map, sigma=5.0)
    atlas_to_pat_avg_label_map = sitk.Cast(sitk.RescaleIntensity(atlas_to_pat_avg_label_map, outputMinimum=0.0, outputMaximum=255.0),
                            pixelID=sitk.sitkUInt8)
    patient.set_image('prob_map', atlas_to_pat_avg_label_map, save_image=True)
    if debug:
        atlas_ct_image = atlas.get_image('ct')
        resampler.SetDefaultPixelValue(-1024)
        atlas_to_pat_ct_image = resampler.Execute(atlas_ct_image)
        patient.set_image('atlas_to_pat_ct', atlas_to_pat_ct_image, save_image=True)


def process_file(filename: str, patient_id: str):
    print("#######################################################################")
    print(f"   PROCESS {patient_id} file {filename}\n")
    st = time.time()
    use_force = True
    use_debug = False
    patient = PatientCase(patient_id, ct_filename=filename)
    total_segmentation(patient, force=use_force, debug=use_debug)
    atlas = Atlas()
    linear_registration(patient, atlas, force=use_force, debug=use_debug)
    resample_images(patient, atlas, force=use_force, debug=use_debug)
    variational_registration(patient, atlas, force=use_force, debug=use_debug)
    transform_atlas_maps_to_patient(patient, atlas, force=use_force, debug=use_debug)
    # crop and save cropped images
    print(f"CROP images to lung mask for patient id {patient.patient_id}")
    lung_mask = patient.get_image('lung_mask')
    ct_image = patient.get_image('ct')
    dist_map = patient.get_image('distance_map')
    prob_map = patient.get_image('prob_map')
    crop_ct_image, crop_dist_map, crop_prob_map = crop_images_to_mask([ct_image, dist_map, prob_map],
                                                                      mask_image=lung_mask)
    patient.set_image('crop_ct', crop_ct_image, save_image=True)
    patient.set_image('crop_distance_map', crop_dist_map, save_image=True)
    patient.set_image('crop_prob_map', crop_prob_map, save_image=True)
    patient.save_image('lung_mask')  # needed for undo cropping in post-processing (or save crop parameters somewhere)
    et = time.time()
    print(f"ELAPSED TIME: {et-st}.")

def process_dir(dirname:str):
    assert os.path.isdir(dirname), f"{dirname} is not a directory!"
    patids = np.unique([filename.split('_')[0] for filename in os.listdir(dirname)])
    for patid in patids:
        ct_filename = glob.glob(os.path.join(dirname, f"{patid}*ct*.nii.gz"))
        process_file(ct_filename[0], patient_id=patid)


if __name__ == '__main__':
    import argparse
    # parse the commandline
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, help='path for input ct image')
    parser.add_argument('--file', '-f', type=str, help='path for input ct image')
    parser.add_argument('--id', '-i', type=str, help='patient id')
    args = parser.parse_args()
    if not (args.dir or (args.file and args.id)):
        parser.print_help()
        exit()
    if args.dir:
        process_dir(args.dir)
    else:
        process_file(args.file, args.id)
