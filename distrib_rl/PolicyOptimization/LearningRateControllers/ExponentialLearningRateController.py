
class ExponentialLearningRateController(object):
    def __init__(self, cfg):
        self.clip_target = cfg["lr_adjuster"].get("clip_target", None)
        self.lr_rate = cfg["lr_adjuster"].get("rate", None)
        self.min_lr = cfg["lr_adjuster"].get("min_lr", 1e-7)
        self.max_lr = cfg["lr_adjuster"].get("max_lr", 1.0)

        if self.lr_rate is None and self.clip_target is not None:
            raise ValueError("Required ExponentialLearningRateController parameter 'rate' is not specified. " +
                "Either specify it or leave 'clip_target' unspecified for no learning rate control.")

    def adjust(self, optimizer, mean_clip):
        if self.clip_target is None:
            return optimizer.step_size

        clip_target = self.clip_target
        rate = self.lr_rate
        min_lr = self.min_lr
        max_lr = self.max_lr
        n = 0
        mean_lr = 0
        lr_report = 0
        if mean_clip > clip_target:
            if hasattr(optimizer, "torch_optimizer"):

                for group in optimizer.torch_optimizer.param_groups:
                    if "lr" in group.keys():
                        group["lr"] /= rate
                        group["lr"] = min(max(group["lr"], min_lr), max_lr)
                        mean_lr += group["lr"]
                        n += 1
                lr_report = mean_lr / n
            else:
                optimizer.step_size /= rate
                optimizer.step_size = min(max(optimizer.step_size, min_lr), max_lr)
                lr_report = optimizer.step_size

        elif mean_clip < clip_target:
            if hasattr(optimizer, "torch_optimizer"):
                for group in optimizer.torch_optimizer.param_groups:
                    if "lr" in group.keys():
                        group["lr"] *= rate
                        group["lr"] = min(max(group["lr"], min_lr), max_lr)
                        mean_lr += group["lr"]
                        n += 1
                lr_report = mean_lr / n

            else:
                optimizer.step_size *= rate
                optimizer.step_size = min(max(optimizer.step_size, min_lr), max_lr)
                lr_report = optimizer.step_size

        return lr_report
