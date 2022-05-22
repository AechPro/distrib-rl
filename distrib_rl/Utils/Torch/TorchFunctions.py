import torch.nn as nn
import torch
import torch.distributions as distribs
import torch.nn.functional as F
import numpy as np


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ClampedLinear(nn.Module):
    def forward(self, x):
        return torch.clamp(x, min=-1, max=1)

class MapContinuousToAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(x)
        n = x.shape[-1] // 2
        
        # condensed version of calling MathHelpers.map_policy_to_action(tanh(x))
        # [0.01, 0.6]
        # return x[..., :n], 0.305 + 0.295 * x[..., n:]

        # [0.01, 1]
        return x[..., :n], 0.55 + 0.45*x[..., n:]

class MultiDiscreteSB3(nn.Module):
    def __init__(self, bins):
        super().__init__()

        self.acts = []
        self.distribution = None
        self.bins = bins
        self.dim = 1

    """Stolen straight from SB3 https://github.com/DLR-RM/stable-baselines3/blob/2bb4500948dccba3292135b1e295532fbc32f668/stable_baselines3/common/distributions.py#L300"""
    def make_distribution(self, policy_output):
        if len(policy_output.shape) == 1:
            self.dim = 0
        else:
            self.dim = 1

        self.distribution = [distribs.Categorical(logits=split) for split in torch.split(policy_output, tuple(self.bins), dim=self.dim)]

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, torch.unbind(actions, dim=self.dim))], dim=self.dim
        ).sum(dim=self.dim)

    def entropy(self) -> torch.Tensor:
        return torch.stack([dist.entropy() for dist in self.distribution], dim=self.dim).sum(dim=self.dim)

    def sample(self) -> torch.Tensor:
        return torch.stack([dist.sample() for dist in self.distribution], dim=self.dim)

class MultiDiscreteRolv(nn.Module):
    def __init__(self, bins):
        super().__init__()
        self.distribution = None
        self.bins = bins

    def make_distribution(self, logits):
        logits = torch.split(logits, self.bins, dim=-1)
        triplets = torch.stack(logits[:5])
        duets = F.pad(torch.stack(logits[5:]), pad=(0, 1), value=float("-inf"))
        logits = torch.cat((triplets, duets)).swapdims(0, 1).squeeze()
        self.distribution = distribs.Categorical(logits=logits)

    def log_prob(self, action):
        return self.distribution.log_prob(action).sum(dim=-1)

    def sample(self):
        return self.distribution.sample()

    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

def init_weights_orthogonal(module: nn.Module, gain: float = 1) -> None:
    """
    Orthogonal initialization (used in PPO and A2C)
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            module.bias.data.fill_(0.0)