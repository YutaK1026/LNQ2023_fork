from torch.optim.lr_scheduler import LRScheduler


class PolyLRScheduler(LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps_bc: int, max_steps_ac: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps_bc = max_steps_bc
        self.max_steps_ac = max_steps_ac
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps_bc) ** self.exponent

        if current_step >= self.max_steps_bc:
            lr_at_change = self.initial_lr * (1 - (self.max_steps_bc - 1)/ self.max_steps_bc) ** self.exponent
            new_lr = lr_at_change * (1 - (current_step - self.max_steps_bc)/ (self.max_steps_ac - self.max_steps_bc)) ** 0.9

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
