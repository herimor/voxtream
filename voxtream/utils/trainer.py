import torch


class LinearWarmupDecayScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        initial_lr: float,
        final_lr: float,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr

        # Precompute learning rates for every step
        self.lrs = torch.cat(
            (
                torch.linspace(initial_lr, final_lr, warmup_steps),
                torch.linspace(final_lr, initial_lr, total_steps - warmup_steps + 1),
            )
        )
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        step = min(step, len(self.lrs) - 1)  # Prevent out-of-bounds
        return [self.lrs[step] for _ in self.base_lrs]
