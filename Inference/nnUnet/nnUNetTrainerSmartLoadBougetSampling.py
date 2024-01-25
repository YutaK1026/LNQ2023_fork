from nnUnet.nnUNetTrainerExtraInfo import nnUNetTrainerExtraInfo
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
import numpy as np
import torch
from nnUnet.smart_cache_nondet_multi_threaded_augmenter import SmartCacheNonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.utilities.helpers import empty_cache
from sklearn.model_selection import KFold
from torch import distributed as dist
from typing import Tuple
import shutil


class nnUNetTrainerSmartLoadBougetSampling(nnUNetTrainerExtraInfo):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_batches_cached = 5
        self.replace_rate = 0.1 # replace all data in 10 epochs

    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            order_resampling_data=3, order_resampling_seg=1,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dl_tr, dl_tr_extra, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_train_extra = SingleThreadedAugmenter(dl_tr_extra, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            # Setup chached loader for training, cache size must be identical to number of iterations
            cache_size = self.num_iterations_per_epoch
            replace_rate = self.replace_rate  
            batch_cache = self.num_batches_cached  # numbers of augmented data cached for batches
            # total needed size is cache_size + (cache_size*replace_rate) + batch_cache
            # args: data_loader, transform, num_processes, cache_size, replace_rate=0.1, num_batch_cached=2, seeds=None, pin_memory=False, wait_time=0.02):
            self.print_to_log_file(f"Training: Use  SmartCacheNonDetMultiThreadedAugmenter(num_processes={allowed_num_processes}, cache_size={cache_size}, replace_rate={replace_rate}, num_batch_cached={batch_cache})")
            mt_gen_train = SmartCacheNonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=tr_transforms,
                                                                  num_processes=allowed_num_processes,
                                                                  cache_size=cache_size, replace_rate=replace_rate,
                                                                  num_batch_cached=batch_cache, seeds=None,
                                                                  pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_train_extra = SmartCacheNonDetMultiThreadedAugmenter(data_loader=dl_tr_extra, transform=tr_transforms,
                                                                  num_processes=allowed_num_processes,
                                                                  cache_size=cache_size, replace_rate=replace_rate,
                                                                  num_batch_cached=batch_cache, seeds=None,
                                                                  pin_memory=self.device.type == 'cuda', wait_time=0.02)

            #mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
            #                               transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
            #                               num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
            #                               wait_time=0.02)
            
            cache_size = self.num_val_iterations_per_epoch
            replace_rate = 0.2  # replace all data in 5 epochs
            batch_cache = max(2, int(self.num_batches_cached/2 + 0.5))   # numbers of augmented data cached for batches
            self.print_to_log_file(f"Validation: Use  SmartCacheNonDetMultiThreadedAugmenter(num_processes={allowed_num_processes}, cache_size={cache_size}, replace_rate={replace_rate}, num_batch_cached={batch_cache})")
            mt_gen_val = SmartCacheNonDetMultiThreadedAugmenter(data_loader=dl_val, transform=val_transforms,
                                                                  num_processes=max(1, allowed_num_processes // 2),
                                                                  cache_size=cache_size, replace_rate=replace_rate,
                                                                  num_batch_cached=batch_cache, seeds=None,
                                                                  pin_memory=self.device.type == 'cuda', wait_time=0.02)
        return mt_gen_train, mt_gen_train_extra, mt_gen_val
        
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        # train dataloader is smart-cache and we have to update the cache
        self.dataloader_train.update_cache()
        self.dataloader_train_extra.update_cache()

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        # validation dataloader is smart-cache and we have to update the cache
        self.dataloader_val.update_cache()

    ############# add another dataloader #############
    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()

        maybe_mkdir_p(self.output_folder)

        # make sure deep supervision is on in the network
        self.set_deep_supervision_enabled(True)

        self.print_plans()
        empty_cache(self.device)

        # maybe unpack
        if self.unpack_dataset and self.local_rank == 0:
            self.print_to_log_file('unpacking dataset...')
            unpack_dataset(self.preprocessed_dataset_folder, unpack_segmentation=True, overwrite_existing=False,
                           num_processes=max(1, round(get_allowed_n_proc_DA() // 2)))
            self.print_to_log_file('unpacking done...')

        if self.is_ddp:
            dist.barrier()

        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_train_extra, self.dataloader_val = self.get_dataloaders()

        # copy plans and dataset.json so that they can be used for restoring everything we need for inference
        save_json(self.plans_manager.plans, join(self.output_folder_base, 'plans.json'), sort_keys=False)
        save_json(self.dataset_json, join(self.output_folder_base, 'dataset.json'), sort_keys=False)

        # we don't really need the fingerprint but its still handy to have it with the others
        shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                    join(self.output_folder_base, 'dataset_fingerprint.json'))

        # produces a pdf in output folder
        self.plot_network_architecture()

        self._save_debug_information()

        # print(f"batch size: {self.batch_size}")
        # print(f"oversample: {self.oversample_foreground_percent}")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_tr_extra, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        else:
            dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_tr_extra = nnUNetDataLoader3D(dataset_tr_extra, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None)
            dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None)
        return dl_tr, dl_tr_extra, dl_val

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, tr_extra_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_tr_extra = nnUNetDataset(self.preprocessed_dataset_folder, tr_extra_keys,
                                   folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_tr_extra, dataset_val

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            case_identifiers = get_case_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = nnUNetDataset(self.preprocessed_dataset_folder, case_identifiers=None,
                                    num_images_properties_loading_threshold=0,
                                    folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append({})
                    splits[-1]['train'] = list(train_keys)
                    splits[-1]['val'] = list(test_keys)
                save_json(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                tr_keys_extra = splits[self.fold]['train_extra']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
        return tr_keys, tr_keys_extra, val_keys
