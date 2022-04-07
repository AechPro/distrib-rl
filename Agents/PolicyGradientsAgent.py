from Agents import BaseAgent
from Utils import MathHelpers as RLMath
import numpy as np
import torch

class PolicyGradientsAgent(BaseAgent):
    def _get_policy_action(self, policy, obs, timestep, evaluate=False):
        inp = torch.as_tensor(obs, dtype=torch.float32)

        if evaluate:
            output = policy.get_output(inp)
            action = output.argmax().item()
        else:
            action, log_prob = policy.get_action(inp)
            timestep.action = action
            timestep.log_prob = log_prob

        return action
