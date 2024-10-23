import inspect
import os
import traceback
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    save_json,
)
from nnUnet.final_trainer import trainer
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.utilities.label_handling.label_handling import (
    determine_num_input_channels,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch._dynamo import OptimizedModule


class predictor(nnUNetPredictor):

    def __init__(
        self,
        tile_step_size: float = 0.5,
        use_gaussian: bool = True,
        use_mirroring: bool = True,
        perform_everything_on_gpu: bool = True,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
        verbose_preprocessing: bool = False,
        allow_tqdm: bool = True,
        ensemble_method: str = "avg",
    ):
        super().__init__(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_gpu=perform_everything_on_gpu,
            device=device,
            verbose=verbose,
            verbose_preprocessing=verbose_preprocessing,
            allow_tqdm=allow_tqdm,
        )
        self.ensemble_method = ensemble_method

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_folds: Union[Tuple[Union[int, str]], None],
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """
        This is used when making predictions with a trained model
        """
        if use_folds is None:
            use_folds = nnUNetPredictor.auto_detect_available_folds(
                model_training_output_dir, checkpoint_name
            )

        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != "all" else f
            checkpoint = torch.load(
                join(model_training_output_dir, f"fold_{f}", checkpoint_name),
                map_location=torch.device("cpu"),
            )
            if i == 0:
                trainer_name = checkpoint["trainer_name"]
                configuration_name = checkpoint["init_args"]["configuration"]
                inference_allowed_mirroring_axes = (
                    checkpoint["inference_allowed_mirroring_axes"]
                    if "inference_allowed_mirroring_axes" in checkpoint.keys()
                    else None
                )

            parameters.append(checkpoint["network_weights"])

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json
        )
        network = trainer.build_network_architecture(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision=False,
        )
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if (
            ("nnUNet_compile" in os.environ.keys())
            and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t"))
            and not isinstance(self.network, OptimizedModule)
        ):
            print("compiling network")
            self.network = torch.compile(self.network)

    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
        # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
        # things a lot faster for some datasets.
        original_perform_everything_on_gpu = self.perform_everything_on_gpu

        list_predictions = []
        with torch.no_grad():
            prediction = None
            if self.perform_everything_on_gpu:
                try:
                    for params in self.list_of_parameters:

                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if prediction is None:
                            prediction = self.predict_sliding_window_return_logits(data)
                            list_predictions.append(prediction)
                        else:
                            list_predictions.append(
                                self.predict_sliding_window_return_logits(data)
                            )

                    if len(self.list_of_parameters) > 1:
                        prediction = list_predictions

                except RuntimeError:
                    print(
                        "Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. "
                        "Falling back to perform_everything_on_gpu=False. Not a big deal, just slower..."
                    )
                    print("Error:")
                    traceback.print_exc()
                    prediction = None
                    self.perform_everything_on_gpu = False

                if prediction is None:

                    for params in self.list_of_parameters:

                        # messing with state dict names...
                        if not isinstance(self.network, OptimizedModule):
                            self.network.load_state_dict(params)
                        else:
                            self.network._orig_mod.load_state_dict(params)

                        if prediction is None:
                            prediction = self.predict_sliding_window_return_logits(data)
                            list_predictions.append(prediction)
                        else:
                            list_predictions.append(
                                self.predict_sliding_window_return_logits(data)
                            )

                    prediction = list_predictions

                print("Prediction done, transferring to CPU if needed")
                if type(prediction) == list:
                    [pred.to("cpu") for pred in prediction]
                else:
                    prediction = prediction.to("cpu")
                self.perform_everything_on_gpu = original_perform_everything_on_gpu
            return prediction

    def predict_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
    ):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(
                output_folder_or_list_of_truncated_output_files[0]
            )
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs
            )  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            maybe_mkdir_p(output_folder)
            save_json(
                my_init_kwargs, join(output_folder, "predict_from_raw_data_args.json")
            )

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(
                self.dataset_json, join(output_folder, "dataset.json"), sort_keys=False
            )
            save_json(
                self.plans_manager.plans,
                join(output_folder, "plans.json"),
                sort_keys=False,
            )
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, (
                f"The requested configuration is a cascaded network. It requires the segmentations of the previous "
                f"stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where"
                f" they are located via folder_with_segs_from_prev_stage"
            )

        # sort out input and output filenames
        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )
        if len(list_of_lists_or_source_folder) == 0:
            return

        preprocessor = self.configuration_manager.preprocessor_class(verbose=False)
        data, seg, properties = preprocessor.run_case(
            list_of_lists_or_source_folder[0],
            None,
            self.plans_manager,
            self.configuration_manager,
            self.dataset_json,
        )

        data = torch.from_numpy(data).contiguous().float()

        prediction = self.predict_logits_from_preprocessed_data(data)

        if type(prediction) == list:
            converted_predictions = []
            for pred in prediction:
                pred = pred.cpu()
                prediction_converted = (
                    convert_predicted_logits_to_segmentation_with_correct_shape(
                        pred,
                        self.plans_manager,
                        self.configuration_manager,
                        self.label_manager,
                        properties,
                        save_probabilities,
                    )
                )
                converted_predictions.append(prediction_converted)
            list_pred = [i[1][1] for i in converted_predictions]
            if self.ensemble_method == "avg":
                prediction = np.mean(np.stack(list_pred), axis=0)
            elif self.ensemble_method == "max":
                prediction = np.max(np.stack(list_pred), axis=0)
            else:
                print("Unknown ensemble method.")
            return prediction, properties
        else:
            prediction = prediction.cpu()
            prediction_converted = (
                convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    self.plans_manager,
                    self.configuration_manager,
                    self.label_manager,
                    properties,
                    save_probabilities,
                )
            )
            return prediction_converted[1][1], properties
