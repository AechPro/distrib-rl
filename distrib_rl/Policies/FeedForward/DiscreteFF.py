import torch
import torch.nn as nn
from torch.distributions import Categorical
from distrib_rl.Policies import Policy
from distrib_rl.Utils.Torch import TorchModelBuilder


class DiscreteFF(Policy):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

    def get_action(self, obs, deterministic=False):
        probs = self.get_output(obs)
        if deterministic:
            return probs.cpu().numpy().argmax(), 0

        distribution = Categorical(probs=probs)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.cpu().item(), log_prob.cpu().item()

    def get_backprop_data(self, obs, acts):
        probs = self.get_output(obs)

        distribution = Categorical(probs=probs)
        entropy = distribution.entropy()
        log_probs = distribution.log_prob(acts)

        return log_probs.to(self.device), entropy.to(self.device).mean()

    def build_model(self, model_json, input_shape, output_shape):
        self.model = TorchModelBuilder.build_from_json(model_json, input_shape, output_shape, channels_first=True)
        self.get_trainable_flat(force_update=True)
        self.model.eval()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._init_params()
