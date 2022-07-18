
def get_from_cfg(cfg, env=None, env_space_shapes=None):
    model_data = {}
    if env is not None:
        output_shape = _get_output_shape(env)
        input_shape = _get_input_shape(env)
    else:
        input_shape, output_shape = env_space_shapes
    device = cfg["device"]

    if "value_estimator" in cfg.keys():
        model_data["value_estimator"] = [
            _get_model_object(cfg["value_estimator"]["type"][0], cfg["value_estimator"]["type"][1]),
            input_shape,
            1, cfg["value_estimator"]]

    if "policy" in cfg.keys():
        model_data["policy"] = [
            _get_model_object(cfg["policy"]["type"][0], cfg["policy"]["type"][1]),
            input_shape,
            output_shape, cfg["policy"]]

    models = {}
    for key, val in model_data.items():
        model_object, in_shape, out_shape, model_json = val
        model = model_object(model_json, device)
        model.build_model(model_json, in_shape, out_shape)
        models[key] = model
    return models


def _get_model_object(policy_type, action_type):
    pt = policy_type.strip().lower()
    at = action_type.strip().lower()
    if pt in ("cnn", "conv", "atari", "convolutional"):
        raise NotImplementedError
        # from distrib_rl.Policies.PyTorch import Convolutional
        # return Convolutional

    if pt in ("rnn", "recurrent", "rec", "lstm", "gru"):
        raise NotImplementedError
        # from distrib_rl.Policies.PyTorch import Recurrent
        # return Recurrent

    else:
        if at == "discrete":
            from distrib_rl.Policies.FeedForward import DiscreteFF
            return DiscreteFF
        elif at == "continuous":
            from distrib_rl.Policies.FeedForward import ContinuousFF
            return ContinuousFF
        elif at == "multi_discrete" or at == "multidiscrete":
            from distrib_rl.Policies.FeedForward import MultiDiscreteFF
            return MultiDiscreteFF

    print("UNABLE TO LOCATE POLICY IMPLEMENTATION MATCHING TYPES: ",policy_type, action_type)
    raise ModuleNotFoundError

def _get_output_shape(env):
    import gym
    a = env.action_space
    if type(a) == gym.spaces.Discrete:
        return a.n
    if type(a) == gym.spaces.Box:
        return a.shape

def _get_input_shape(env):
    return env.observation_space.shape
