from distrib_rl.policies import Policy
from distrib_rl.utils.torch import torch_model_builder, torch_functions


class MultiDiscreteFF(Policy):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.multi_discrete = None
        self.splits = (3,) * 5 + (2,) * 3

    def get_action(self, obs, deterministic=False):
        obs = [obs]
        logits = self.get_output(obs)

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.cpu().numpy(), log_prob.cpu().item()

    def get_backprop_data(self, obs, acts):
        logits = self.get_output(obs)

        distribution = self.multi_discrete
        distribution.make_distribution(logits)

        entropy = distribution.entropy().to(self.device)
        log_probs = distribution.log_prob(acts).to(self.device)

        return log_probs, entropy.mean()

    def build_model(self, model_json, input_shape, output_shape):
        self.model = torch_model_builder.build_from_json(
            model_json, input_shape, output_shape, channels_first=True
        )
        self.get_trainable_flat(force_update=True)
        self.model.eval()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self._init_params()

        bins = model_json["layers"]["output"]["extra"]
        # self.multi_discrete = torch_functions.MultiDiscreteSB3(bins)
        self.multi_discrete = torch_functions.MultiDiscreteRolv(bins)
