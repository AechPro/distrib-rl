

def get_from_cfg(cfg):
    t = cfg["agent"]["type"].lower().strip()
    if t == "pg":
        from Agents import PolicyGradientsAgent
        return PolicyGradientsAgent(cfg)

    if t == "marl":
        from Agents import MARLAgent
        return MARLAgent(cfg)

    print("UNABLE TO LOCATE GRADIENT OPTIMIZER TYPE", t)
    raise ModuleNotFoundError