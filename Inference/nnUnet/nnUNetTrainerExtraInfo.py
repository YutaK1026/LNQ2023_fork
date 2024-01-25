from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch
from torch import autocast


class nnUNetTrainerExtraInfo(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.print_to_log_file(f"Current learning rate: {self.initial_lr}")
        self.print_to_log_file(f"Epochs: start={self.current_epoch}  end={self.num_epochs}")
        self.print_to_log_file(f"Steps per Epoch: {self.num_iterations_per_epoch}")

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad()
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            #for idx in range(len(output)):
            #    print(f"pred {idx} [{torch.min(output[idx]).item()} - {torch.max(output[idx]).item()}] tar [{torch.min(target[idx]).item()} - {torch.max(target[idx]).item()}]")
            # del data
            l = self.loss(output, target)
            # Compute FPR and FNR
            p0, g0 = torch.nn.functional.softmax(output[0], dim=1)[:,1:].detach(), target[0][:].detach()
            p1, g1 = 1 - p0, 1 - g0
            tp = torch.sum(p0 * g0, (2, 3, 4))
            tn = torch.sum(p1 * g1, (2, 3, 4))
            fp = torch.sum(p0 * g1, (2, 3, 4))
            fn = torch.sum(p1 * g0, (2, 3, 4))
            fnr = (fn + 1e-5)/(fn+tp + 1e-5)
            fpr = (fp + 1e-5)/(fp+tn + 1e-5)
            # print(torch.mean(fnr).item(), torch.mean(fpr).item())
            # print(l)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy(), 'FNR': torch.mean(fnr).item(), 'FPR': torch.mean(fpr).item()}

    def on_train_epoch_end(self, train_outputs):
        fpr = np.mean([out['FPR'] for out in train_outputs])
        fnr = np.mean([out['FNR'] for out in train_outputs])
        self.print_to_log_file(f"TRAIN Epoch: {self.current_epoch}, FPR: {fpr}, FNR: {fnr}")
        return super().on_train_epoch_end(train_outputs)

    def on_validation_epoch_end(self, val_outputs):
        fp = np.sum([out['fp_hard'] for out in val_outputs])
        fn = np.sum([out['fn_hard'] for out in val_outputs])
        self.print_to_log_file(f"VAL Epoch: {self.current_epoch}, #FP: {int(fp)}, #FN: {int(fn)}")
        return super().on_validation_epoch_end(val_outputs)