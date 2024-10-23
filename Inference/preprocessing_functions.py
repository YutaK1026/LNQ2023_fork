from pathlib import Path

import SimpleITK as sitk


def from_nrrd_to_nifti(input_path: Path, output_path: Path):
    """Convert from nrrd-files to nifti and rename according to patient id."""
    print("Start: convert nrrd to nifti.")
    patient = input_path.name
    img = sitk.ReadImage(input_path)
    sitk.WriteImage(img, output_path)
    print(f"{patient} is done.")
