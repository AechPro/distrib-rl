import torch
import numpy as np


class REINFORCE(object):
    def __init__(self, cfg, policy, policy_optimizer, gradient_builder, adaptive_omega):
        self.device = cfg["device"]
        self.cfg = cfg
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.gradient_builder = gradient_builder

    def learn(self, exp):
        policy = self.policy
        builder = self.gradient_builder
        optimizer = self.policy_optimizer
        epochs = self.cfg["policy_optimizer"]["n_epochs"]

        n_updates = 0
        mean_entropy = 0
        mean_divergence = 0
        clip_fractions = []
        batches = exp.get_all_batches_shuffled()

        for batch in batches:
            acts = batch.actions
            obs = batch.obs
            rews = batch.future_rewards
            old_probs = batch.log_probs

            log_probs, entropy = policy.get_backprop_data(obs, acts)
            policy_loss = -(rews * log_probs).mean()

            loss = policy_loss
            loss.backward()

            builder.contribute_gradient_from_model(policy, 1.0)
            builder.update_model(policy, optimizer)
            n_updates += 1

            with torch.no_grad():
                log_ratio = log_probs - old_probs
                kl = (torch.exp(log_ratio) - 1) - log_ratio
                kl = kl.mean().detach().cpu().item()

                # From the stable-baselines3 implementation of PPO.
                clip_fractions.append(0)
            mean_divergence += kl
            mean_entropy += entropy.detach().item()

        mean_entropy /= len(batches) * epochs
        mean_divergence /= len(batches) * epochs

        report = {
            "n_updates": n_updates,
            "mean_entropy": mean_entropy,
            "mean_kl": mean_divergence,
            "clip_fraction": np.mean(clip_fractions),
        }

        return report
