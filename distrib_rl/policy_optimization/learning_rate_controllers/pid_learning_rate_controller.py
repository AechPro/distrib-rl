from weakref import ref


class PIDLearningRateController(object):
    def __init__(self, cfg):
        self.clip_target = cfg["lr_adjuster"].get("clip_target", None)
        self.lr_kp = cfg["lr_adjuster"].get("Kp", None)
        self.lr_ki = cfg["lr_adjuster"].get("Ki", 0)
        self.lr_kd = cfg["lr_adjuster"].get("Kd", 0)
        self.min_lr = cfg["lr_adjuster"].get("min_lr", 1e-7)
        self.max_lr = cfg["lr_adjuster"].get("max_lr", 1.0)

        self.state = {}

        if self.lr_kp is None and self.clip_target is not None:
            raise ValueError(
                "Required PIDLearningRateController parameter 'Kp' is not specified. "
                + "Either specify it or leave 'clip_target' unspecified for no learning rate control."
            )

    def adjust(self, optimizer, mean_clip):
        if self.clip_target is None:
            return optimizer.step_size

        # we use a weak reference to the optimizer to avoid holding a
        # reference in the state dict that causes the optimizer to never
        # be gc'd
        po_ref = ref(optimizer)
        last_error = self.state[po_ref].get("last_error", 0)
        integral = self.state[po_ref].get("integral", 0)

        clip_target = self.clip_target

        Kp = self.lr_kp
        Ki = self.lr_ki
        Kd = self.lr_kd

        min_lr = self.min_lr
        max_lr = self.max_lr

        mean_lr = 0

        error = clip_target - mean_clip

        proportional = error * Kp

        derivative = (error - last_error) * Kd if Kd is not None else 0

        integral += error * Ki if Ki is not None else 0

        adjustment = proportional + derivative + integral

        self.state[po_ref]["last_error"] = error
        self.state[po_ref]["integral"] = integral

        if hasattr(optimizer, "torch_optimizer"):
            mean_lr = 0
            n = 0
            for group in optimizer.torch_optimizer.param_groups:
                if "lr" in group.keys():
                    group["lr"] = min(max(group["lr"] + adjustment, min_lr), max_lr)
                    mean_lr += group["lr"]
                    n += 1
            return mean_lr / n
        else:
            optimizer.step_size = min(
                max(optimizer.step_size + adjustment, min_lr), max_lr
            )
            return optimizer.step_size
