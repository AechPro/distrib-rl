

def get_from_cfg(cfg, policy):
    device = cfg["device"]
    optimizers = {}
    for key, val in cfg.items():
        if "gradient_optimizer" in key:
            optimizers[key] = _build_optimizer(val, policy, device)
    return optimizers

def _build_optimizer(cfg_section, policy, device):
    t = cfg_section["type"].lower().strip()
    kwargs = cfg_section.copy()
    del kwargs["type"]

    if t == "dsgd":
        from distrib_rl.GradientOptimization.Optimizers import DynamicSGD
        return DynamicSGD(policy, **kwargs)

    elif t == "adam":
        from distrib_rl.GradientOptimization.Optimizers import Adam
        return Adam(policy, **kwargs)

    elif "torch" in t:
        if "step_size" in kwargs.keys():
            kwargs["lr"] = kwargs["step_size"]
            del kwargs["step_size"]

        from distrib_rl.GradientOptimization.Optimizers import TorchWrapper
        t = t.split(" ")[1]
        if t == "adam":
            from torch.optim import Adam
            optim = Adam(policy.parameters(), **kwargs)
        elif t == "sgd":
            from torch.optim import SGD
            optim = SGD(policy.parameters(), **kwargs)
        elif t == "rmsprop":
            from torch.optim import RMSprop
            optim = RMSprop(policy.parameters(), **kwargs)
        else:
            optim = None

        if optim is None:
            print("UNABLE TO LOCATE GRADIENT OPTIMIZER TYPE", t)
            raise ModuleNotFoundError

        wrapper = TorchWrapper(policy, optim, device)
        return wrapper
