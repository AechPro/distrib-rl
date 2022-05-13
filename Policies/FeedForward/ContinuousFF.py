import torch
from torch.distributions import Normal
from Policies import Policy
from Utils.Torch import TorchModelBuilder
from Utils import MathHelpers as RLMath
import numpy as np
import functools


class ContinuousFF(Policy):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.init_log_std = 0
        self.std = None
        self.step = 0
        self.entropy_min = 0
        self.entropy_max = 1


    @functools.lru_cache()
    def logpdf(self, x, mean, std):
        msq = mean * mean
        ssq = std * std
        xsq = x * x

        term1 = -torch.divide(msq, (2 * ssq))
        term2 = torch.divide(mean * x, ssq)
        term3 = -torch.divide(xsq, (2 * ssq))
        term4 = torch.log(1 / torch.sqrt(2 * np.pi * ssq))

        return term1 + term2 + term3 + term4


    def get_action(self, obs, summed_probs=True, deterministic=False):
        mean, std = self.get_output(obs)
        if deterministic:
            return mean, 0

        # mean, std = RLMath.map_policy_to_continuous_action(model_out)
        # print(mean, std)
        distribution = Normal(loc=mean, scale=std)
        action = distribution.sample()
        log_prob = self.logpdf(action, mean, std)

        shape = log_prob.shape
        if summed_probs:
            if len(shape) > 1:
                log_prob = log_prob.sum(dim=1)
            else:
                log_prob = log_prob.sum()
            log_prob = log_prob.cpu().item()
        else:
            log_prob = log_prob.cpu().numpy()

        return action.cpu().numpy(), log_prob

    def get_backprop_data(self, obs, acts, summed_probs=True):
        mean, std = self.get_output(obs)
        # mean, std = RLMath.map_policy_to_continuous_action(model_out)

        distribution = Normal(loc=mean, scale=std)

        prob = self.logpdf(acts, mean, std)
        if summed_probs:
            log_probs = prob.sum(dim=1).to(self.device)
        else:
            log_probs = prob.to(self.device)

        entropy = distribution.entropy()
        entropy = RLMath.minmax_norm(entropy.sum(dim=1), self.entropy_min, self.entropy_max)
        entropy = entropy.mean().to(self.device)

        return log_probs, entropy

    def build_model(self, model_json, input_shape, output_shape):
        self.model = TorchModelBuilder.build_from_json(model_json, input_shape, output_shape, channels_first=True)

        with torch.no_grad():
            min_sigma = torch.ones(output_shape) * 1e-1
            max_sigma = torch.ones(output_shape)
            self.entropy_min = RLMath.compute_torch_normal_entropy(min_sigma)
            self.entropy_max = RLMath.compute_torch_normal_entropy(max_sigma)
            print("CONFIGURING CONTINUOUS POLICY TO SHIFT ENTROPY VALUES BETWEEN 0 AND 1")
            print("OUTPUT SHAPE:", output_shape, "ENTROPY MIN:", self.entropy_min, "ENTROPY MAX:", self.entropy_max)


        self.model.eval()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._init_params()
        self.get_trainable_flat(force_update=True)
