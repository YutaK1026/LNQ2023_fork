import time

import SimpleITK
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator
from post_processing_functions import *
from preprocessing_functions import *
from preregistration.preregistration import process_file

os.environ["nnUNet_raw"] = "./nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "./nnUNet_preprocessed"
os.environ["nnUNet_results"] = "./weights/nnUNet_results"

from predictor import predictor as nnUNet_predictor

nnUNet_raw_dir = os.environ["nnUNet_raw"]
nnUNet_preprocessed_dir = os.environ["nnUNet_preprocessed"]
nnUNet_results_dir = os.environ["nnUNet_results"]


class Lnq2023(SegmentationAlgorithm):
    def __init__(self):
        output_path = Path("./output/images/mediastinal-lymph-node-segmentation/")
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if not Path(nnUNet_raw_dir).exists():
            Path(nnUNet_raw_dir).mkdir(parents=True, exist_ok=True)
        if not Path(nnUNet_preprocessed_dir).exists():
            Path(nnUNet_preprocessed_dir).mkdir(parents=True, exist_ok=True)
        super().__init__(
            input_path=Path("./input/images/mediastinal-ct/"),
            output_path=output_path,
            validators=dict(input_image=(UniqueImagesValidator(),)),
        )
        self.predictor = nnUNet_predictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_gpu=True,
            device=torch.device("cuda", 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
            ensemble_method="avg",
        )
        self.predictor.initialize_from_trained_model_folder(
            model_training_output_dir=os.path.join(
                nnUNet_results_dir,
                "Dataset017_LNQ2023/trainer_nnUNet_Plans_3d_fullres/",
            ),
            use_folds=tuple([0, 1, 2, 3, 4]),
            checkpoint_name="checkpoint_best.pth",
        )

    def process_case(self, *, idx, case):
        st = time.time()

        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)

        segmented_nodules = self.preprocess(idx, input_image_file_path)

        # Write resulting segmentation to output location
        segmentation_path = self._output_path / input_image_file_path.name
        if not self._output_path.exists():
            self._output_path.mkdir()
        SimpleITK.WriteImage(segmented_nodules, str(segmentation_path), True)

        et = time.time()
        print(f"TOTAL TIME for processing patient {idx}: {et - st}.")

        # Write segmentation file path to result.json for this case
        return {
            "outputs": [dict(type="metaio_image", filename=segmentation_path.name)],
            "inputs": [dict(type="metaio_image", filename=input_image_file_path.name)],
            "error_messages": [],
        }

    def preprocess(self, idx: int, input_image_file_path: Path) -> torch.Tensor:
        name_ct = f"ct_{idx + 1:03d}_0000.nii.gz"
        Path(os.path.join(nnUNet_raw_dir, f"ct_{idx + 1:03d}")).mkdir(exist_ok=True)
        ct_path = os.path.join(nnUNet_raw_dir, f"ct_{idx+1:03d}", name_ct)
        from_nrrd_to_nifti(input_image_file_path, Path(ct_path))
        process_file(ct_path, f"ct_{idx + 1:03d}")
        lungmasks = Path(f"./Results2/subjects/ct_{idx + 1:03d}/")
        name_lungmask = f"ct_{idx+1:03d}_lung_mask.nii.gz"
        lungmask_path = os.path.join(lungmasks, name_lungmask)
        regmask_path = create_regmask(lungmask_path)
        probm_path = os.path.join(
            nnUNet_raw_dir, f"ct_{idx+1:03d}", f"ct_{idx+1:03d}_0002.nii.gz"
        )
        prediction, properties = self.predictor.predict_from_files(
            str(os.path.join(nnUNet_raw_dir, f"ct_{idx + 1:03d}")),
            str(os.path.join(nnUNet_raw_dir, f"ct_{idx + 1:03d}")),
            save_probabilities=True,
            overwrite=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )
        final_predictions_itk = cut_predictions(prediction, probm_path, 0.3)
        final_predictions_itk.SetOrigin(properties["sitk_stuff"]["origin"])
        final_predictions_itk.SetSpacing(properties["sitk_stuff"]["spacing"])
        final_predictions_itk.SetDirection(properties["sitk_stuff"]["direction"])
        regmask_img = sitk.ReadImage(regmask_path)
        label_img = pad_to_original_size(final_predictions_itk, regmask_img)
        label_img = remove_outside_lung(label_img, regmask_img)
        label_img = remove_healthy_lymphnodes_sitk(label_img, 5)
        label_img = sitk.Cast(label_img, sitk.sitkUInt8)
        return label_img

    def predict(self, *, input_image: SimpleITK.Image) -> SimpleITK.Image:
        raise NotImplementedError()


if __name__ == "__main__":
    Lnq2023().process()
