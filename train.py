import json
import os
import argparse

os.environ['nnUNet_n_proc_DA'] = '20'

from Inference.nnUnet.final_trainer import trainer

nnUNet_raw_dir = os.environ['nnUNet_raw']
nnUNet_preprocessed_dir = os.environ['nnUNet_preprocessed']
nnUNet_results_dir = os.environ['nnUNet_results']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='e.g. Dataset017_LNQ2023')
    parser.add_argument('--fold', type=int, help='e.g. 1')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='e.g. checkpoint_latest.pth')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    plans = json.load(open(f'{nnUNet_preprocessed_dir}/{args.dataset_name}/nnUNetPlans.json'))
    configuration = '3d_fullres'
    dataset_json = json.load(open(f'{nnUNet_preprocessed_dir}/{args.dataset_name}/dataset.json'))
    plans['max_epochs_lr_decay'] = 1000
    plans['num_epochs'] = 3000
    plans['use_random_conv_until'] = 3000
    plans['saturation_epoch'] = 1000
    print(plans)

    trainer_class = trainer(plans, configuration, args.fold, dataset_json)
    trainer_class.initialize()
    if args.checkpoint_name:
        checkpoint_file = f'{nnUNet_results_dir}/{args.dataset_name}/trainer__nnUNetPlans__3d_fullres/fold_{args.fold}/{args.checkpoint_name}'
        trainer_class.load_checkpoint(checkpoint_file)

    #
    # Print network structure and parameters
    #
    old_folder = trainer_class.output_folder
    trainer_class.output_folder = "./"
    trainer_class.plot_network_architecture()
    trainer_class.output_folder = old_folder
    network = trainer_class.network
    for name, param in network.named_parameters():
        print(name, param.shape)

    #
    # Run Training
    #
    trainer_class.print_to_log_file(f"Epochs: start={trainer_class.current_epoch}  end={trainer_class.num_epochs}")
    trainer_class.print_to_log_file(f"Steps per Epoch: {trainer_class.num_iterations_per_epoch}")
    trainer_class.run_training()

    exit()
