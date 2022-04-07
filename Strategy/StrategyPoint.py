import torch
import numpy as np
from Policies.FeedForward import ContinuousFF

class StrategyPoint(object):
    def __init__(self, flat):
        self.flat = flat
        self.strategy = None

    @torch.no_grad()
    def compute_strategy(self, policy, frames):
        orig = policy.get_trainable_flat()
        policy.set_trainable_flat(self.flat)
        self.strategy = policy.get_output(frames).cpu().flatten().numpy().astype(np.float32)
        policy.set_trainable_flat(orig)

    def cleanup(self):
        del self.flat
        del self.strategy