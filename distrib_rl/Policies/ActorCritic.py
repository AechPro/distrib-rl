from distrib_rl.Policies import Policy
import torch.nn as nn


class CombinedModel(nn.Module):
    def __init__(self, policy, value_net):
        super().__init__()
        self.actor = policy
        self.critic = value_net


class ActorCritic(Policy):
    def __init__(self, policy, value_net, cfg):
        super().__init__(cfg)
        self.model = CombinedModel(policy, value_net)
        self.num_params = len(policy.get_trainable_flat(force_update=True)) + len(
            value_net.get_trainable_flat(force_update=True)
        )

    def get_backprop_data(self, obs, acts):
        pass

    def get_action(self, obs):
        pass

    def build_model(self, model_json, input_shape, output_shape):
        pass

    def set_device(self, device):
        self.model.actor.to(device)
        self.model.critic.to(device)
        self.to(device)
