import os

import SimpleITK as sitk
from preregistration.prereg_utils import undo_resample_image_to_2x2x2

BASEPATH = "/opt/app/"
nnUNet_raw = os.path.join(BASEPATH, "nnUNet_raw")
ATLASPATH = os.path.join(BASEPATH, "weights", "data", "Atlas")
RESULTPATH = os.path.join(BASEPATH, "Results2")
PNGPATH = os.path.join(RESULTPATH, "PNGs")


def ensure_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=False)
    assert os.path.isdir(path), f"{path} is not a directory!"


class PatientCase(object):
    def __init__(
        self, patient_id, sitk_ct_image=None, ct_filename=None, force=False, debug=True
    ):
        assert (sitk_ct_image is None) != (
            ct_filename is None
        ), f"Patient case {patient_id}: You have to provide a ct image XOR a ct filename!"
        self.patient_id = patient_id
        self.debug = debug
        self.output_dir = os.path.join(RESULTPATH, "subjects", self.patient_id)
        ensure_path(self.output_dir)
        self.png_dir = PNGPATH  # only for debug

        # list of images that should be available/generated per patient
        self.image_keys = [
            "ct",
            "total_segm",
            "reg_mask",
            "lung_mask",
            "ct_rigid",
            "mask_rigid",
            "transform_rigid",
            "ct_affine",
            "mask_affine",
            "transform_affine",
            "ct_affine_resample",
            "segm_affine_resample",
            "to_atlas_displ",
            "to_atlas_velo",
            "to_atlas_ct_warped",
            "to_atlas_segm_warped",
            "distance_map",
            "prob_map",
            "atlas_to_pat_ct",
            "crop_ct",
            "crop_distance_map",
            "crop_prob_map",
        ]
        self.filenames = {key: None for key in self.image_keys}
        self.images = {key: None for key in self.image_keys}
        self._setup_filenames()  # setup default file names
        self.filenames["ct"] = os.path.join(
            nnUNet_raw, patient_id, f"{patient_id}_0000.nii.gz"
        )
        self.filenames["crop_ct"] = os.path.join(
            nnUNet_raw, patient_id, f"{patient_id}_0000.nii.gz"
        )
        self.filenames["crop_distance_map"] = os.path.join(
            nnUNet_raw, patient_id, f"{patient_id}_0001.nii.gz"
        )
        self.filenames["crop_prob_map"] = os.path.join(
            nnUNet_raw, patient_id, f"{patient_id}_0002.nii.gz"
        )
        if sitk_ct_image is not None:
            self.images["ct"] = sitk_ct_image

    def _setup_filenames(self):
        # make one directory containing all files per patient
        for key in self.image_keys:  # setup default for all images
            self.filenames[key] = os.path.join(
                self.output_dir, f"{self.patient_id}_{key}.nii.gz"
            )
        for key in ["transform_rigid", "transform_affine"]:  # correct for transforms
            self.filenames[key] = os.path.join(
                self.output_dir, f"atlas_to_{self.patient_id}_{key}.tfm"
            )

    def _load_image(self, key: str):
        assert (
            key in self.image_keys
        ), f"Patient case {self.patient_id}: key {key} is not valid!"
        if not self.filenames[key]:
            print(
                f"ERROR Patient case {self.patient_id}: Filename for key {key} not set!"
            )
            return False
        if not os.path.isfile(self.filenames[key]):
            print(
                f"ERROR Patient case {self.patient_id}: File for key {key} in {self.filenames[key]} does not exist!"
            )
            return False
        if self.debug:
            print(
                f"case {self.patient_id}: LOAD image {key} from {self.filenames[key]}"
            )
        if key in ["transform_rigid", "transform_affine"]:
            self.images[key] = sitk.ReadTransform(self.filenames[key])
        else:
            self.images[key] = sitk.ReadImage(self.filenames[key])
        return True

    def is_available(self, key: str):
        assert (
            key in self.image_keys
        ), f"Patient case {self.patient_id}: key {key} is not valid!"
        return self.images[key] is not None or (
            self.filenames[key] and os.path.isfile(self.filenames[key])
        )

    def get_image(self, key: str, as_pixel_type=None):
        assert (
            key in self.image_keys
        ), f"Patient case {self.patient_id}: key {key} is not valid!"
        if self.images[key] is None:
            self._load_image(key)
        if as_pixel_type is not None:
            return sitk.Cast(self.images[key], pixelID=as_pixel_type)
        return self.images[key]

    def set_image(self, key: str, image: sitk.Image, save_image=False):
        assert (
            key in self.image_keys
        ), f"Patient case {self.patient_id}: key {key} is not valid!"
        self.images[key] = image
        if save_image and image is not None:
            self.save_image(key)

    def save_image(self, key: str):
        assert (
            key in self.image_keys
        ), f"Patient case {self.patient_id}: key {key} is not valid!"
        if self.images[key] is None:
            print(f"ERROR Patient case {self.patient_id}: Image for key {key} not set!")
            return False
        if not self.filenames[key]:
            print(
                f"ERROR Patient case {self.patient_id}: Filename for key {key} not set!"
            )
            return False
        if self.debug:
            print(f"case {self.patient_id}: SAVE image {key} to {self.filenames[key]}")
        if key in ["transform_rigid", "transform_affine"]:
            sitk.WriteTransform(self.images[key], self.filenames[key])
        else:
            sitk.WriteImage(self.images[key], self.filenames[key])
        return True


class Atlas(PatientCase):
    def __init__(self, debug=True):
        super().__init__(
            patient_id="atlas2",
            ct_filename=os.path.join(ATLASPATH, "patient2-ct.nii.gz"),
            debug=debug,
        )
        self.output_dir = ATLASPATH  # abused for input dir
        # now remove keys and create new ones
        self.image_keys = [
            "ct",
            "total_segm",
            "reg_mask",
            "ct_resample",
            "segm_resample",
            "average_labels",
        ]
        self.filenames = {key: None for key in self.image_keys}
        self.images = {key: None for key in self.image_keys}
        # add filenames
        self.filenames["ct"] = os.path.join(ATLASPATH, "patient2-ct.nii.gz")
        self.filenames["total_segm"] = os.path.join(
            ATLASPATH, "atlas2_total_segm.nii.gz"
        )
        self.filenames["reg_mask"] = os.path.join(ATLASPATH, "atlas2_reg_mask.nii.gz")
        self.filenames["ct_resample"] = os.path.join(
            ATLASPATH, "atlas2_ct_affine_resample.nii.gz"
        )
        self.filenames["segm_resample"] = os.path.join(
            ATLASPATH, "atlas2_segm_affine_resample.nii.gz"
        )
        self.filenames["average_labels"] = os.path.join(
            ATLASPATH, "average_labels_from_LNQ_train_scaled.nii.gz"
        )
        # build index map
        self.atlas_coordinate_map = self.compute_template_coordinate_map(
            self.get_image("ct_resample"), ref_index=[80, 59, 90]
        )
        self.coord_map_bg_value = float(
            sitk.GetArrayFromImage(self.atlas_coordinate_map).max()
        )  # max coord as bg value
        orig_size_atlas_image = self.get_image(
            "total_segm"
        )  # faster to load (and needed anyway) as self.get_image('ct')
        self.atlas_coordinate_map = undo_resample_image_to_2x2x2(
            self.atlas_coordinate_map,
            orig_size_atlas_image,
            interpolator=sitk.sitkLinear,
            default_value=self.coord_map_bg_value,
        )

    def save_image(self, key: str):
        assert False, "Never overwrite Atlas images!"

    def compute_template_coordinate_map(self, image, ref_index):
        ref_point = image.TransformIndexToPhysicalPoint(ref_index)
        # print(ref_index, ref_point)
        coordinate_map = sitk.PhysicalPointSource(
            sitk.sitkVectorFloat32,
            size=image.GetSize(),
            origin=image.GetOrigin(),
            spacing=image.GetSpacing(),
            direction=image.GetDirection(),
        )
        imgx = sitk.VectorIndexSelectionCast(coordinate_map, 0) - ref_point[0]
        imgy = sitk.VectorIndexSelectionCast(coordinate_map, 1) - ref_point[1]
        imgz = sitk.VectorIndexSelectionCast(coordinate_map, 2) - ref_point[2]
        # magnitude = sitk.Sqrt(imgx**2 + imgy**2 + imgz**2)
        coordinate_map = sitk.Compose([imgx, imgy, imgz])
        return coordinate_map
