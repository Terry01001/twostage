from torch.optim.lr_scheduler import _LRScheduler, StepLR
import torch


def get_scheduler(opts, optim):
    if opts.lr_policy == 'poly':
        scheduler = PolyLR(optim, max_iters=opts.max_iters, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opts.lr_decay_step,
                                                    gamma=opts.lr_decay_factor)
    elif opts.lr_policy == 'polyLR':
        scheduler = PolynomialLR(optim, step_size=1, iter_warmup=0, 
                                 iter_max=opts.max_iters, power=opts.lr_power)
    elif opts.lr_policy == 'warmup':
        scheduler = WarmUpPolyLR(optim, max_iters=opts.max_iters, power=opts.lr_power, start_decay=opts.start_decay)
    elif opts.lr_policy == 'none':
        scheduler = NoScheduler(optim)
    else:
        raise NotImplementedError
    return scheduler

class PolynomialLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        step_size,
        iter_warmup,
        iter_max,
        power,
        min_lr=1e-5,
        last_epoch=-1,
    ):
        self.step_size = step_size
        self.iter_warmup = int(iter_warmup)
        self.iter_max = int(iter_max)
        self.power = power
        self.min_lr = min_lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        iter_cur = float(self.last_epoch)
        if iter_cur < self.iter_warmup:
            coef = iter_cur / self.iter_warmup
            coef *= (1 - self.iter_warmup / self.iter_max) ** self.power
        else:
            coef = (1 - iter_cur / self.iter_max) ** self.power
        return (lr - self.min_lr) * coef + self.min_lr

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]

    def step_update(self, num_updates):
        self.step()


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
                for base_lr in self.base_lrs]


class NoScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(NoScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


class WarmUpPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, start_decay=20, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        self.start_decay = start_decay
        super(WarmUpPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.start_decay:
            return [base_lr * (1 - self.last_epoch / self.max_iters) ** self.power
                    for base_lr in self.base_lrs]
        else:
            return self.base_lrs
