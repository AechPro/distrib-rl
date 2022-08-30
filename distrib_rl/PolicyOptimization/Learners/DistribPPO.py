import torch
import time
import numpy as np
from distrib_rl.PolicyOptimization.LearningRateControllers.ExponentialLearningRateController import ExponentialLearningRateController

from distrib_rl.PolicyOptimization.LearningRateControllers.PIDLearningRateController import PIDLearningRateController


class DistribPPO(object):
    def __init__(self, cfg, policy, value_net, policy_optimizer, value_optimizer, gradient_builder, adaptive_omega):
        self.cfg = cfg
        self.device = cfg["device"]
        self.policy = policy
        self.policy_optimizer = policy_optimizer
        self.value_net = value_net
        self.value_optimizer = value_optimizer
        self.gradient_builder = gradient_builder
        self.adaptive_omega = adaptive_omega
        self.value_loss_fn = torch.nn.MSELoss()
        self.last_error = None
        self.adjust_lr_every_batch = False

        if "lr_adjuster" in cfg:
            if "Kp" in cfg["lr_adjuster"]:
                self.lr_adjuster = PIDLearningRateController(cfg)
                self.lr_adjust_every_batch = True
            elif "rate" in cfg["lr_adjuster"]:
                self.lr_adjuster = ExponentialLearningRateController(cfg)
            else:
                self.lr_adjuster = None
        else:
            self.lr_adjuster = None

        self.is_torch_optimizer = "torch" in cfg["policy_gradient_optimizer"]["type"]

        self.n_epochs = 0
        self.cumulative_model_updates = 0
        self.burn_in = 0

    def learn(self, exp):
        policy = self.policy
        value_net = self.value_net
        policy_optimizer = self.policy_optimizer
        value_optimizer = self.value_optimizer
        val_loss_fn = self.value_loss_fn
        builder = self.gradient_builder
        clip_range = self.cfg["policy_optimizer"]["clip_range"]
        ent_coef = self.cfg["policy_optimizer"]["entropy_coef"]
        device = self.device
        n_epochs = self.n_epochs
        burn_in = self.burn_in

        n_updates = 0
        n_iterations = 0
        mean_entropy = 0
        mean_divergence = 0
        mean_val_loss = 0
        clip_fractions = []
        lr_report = 0

        batches = exp.get_all_batches_shuffled()

        t1 = time.time()
        for batch in batches:
            acts, old_probs, obs, target_values, advantages = batch

            acts = acts.to(device)
            obs = obs.to(device)
            advantages = advantages.to(device)
            old_probs = old_probs.to(device)
            target_values = target_values.to(device)

            vals = value_net.get_output(obs).view_as(target_values)

            value_loss = val_loss_fn(vals, target_values)
            if n_epochs >= burn_in:
                log_probs, entropy = policy.get_backprop_data(obs, acts)
                ratio = torch.exp(log_probs - old_probs)
                clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -torch.min(ratio*advantages, clipped*advantages).mean()
                with torch.no_grad():
                    log_ratio = log_probs - old_probs
                    kl = (torch.exp(log_ratio) - 1) - log_ratio
                    kl = kl.mean().detach().cpu().item()

                    # From the stable-baselines3 implementation of PPO.
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.lr_adjuster is not None and self.adjust_lr_every_batch:
                        lr_report += self.lr_adjuster.adjust(self.policy_optimizer, clip_fraction)

                loss = policy_loss - entropy * ent_coef
                loss.backward()
                builder.contribute_gradient_from_model(policy, 0.5, force_norm=False)
                builder.update_model(policy, policy_optimizer)
                n_updates += 1

            else:
                policy_loss = torch.zeros_like(value_loss)
                entropy = torch.zeros_like(value_loss)
                policy_loss.to(device)
                entropy.to(device)
                clip_fractions.append(0)
                kl = 0

            mean_val_loss += value_loss.detach().item()
            mean_divergence += kl
            mean_entropy += entropy.detach().item()

            loss = value_loss
            loss.backward()
            builder.contribute_gradient_from_model(value_net, 1.0, force_norm=False)
            builder.update_model(value_net, value_optimizer)

            n_iterations += 1

        if n_iterations == 0:
            n_iterations = 1

        mean_entropy /= n_iterations
        mean_divergence /= n_iterations
        mean_val_loss /= n_iterations
        mean_clip = np.mean(clip_fractions)

        if self.adjust_lr_every_batch:
            lr_report /= n_iterations
        elif self.lr_adjuster is not None:
            lr_report = self.lr_adjuster.adjust(self.policy_optimizer, mean_clip)


        self.n_epochs += 1
        self.cumulative_model_updates += n_updates
        report = {
                 "batch_time": (time.time() - t1) / n_iterations,
                 "n_batches" : n_iterations,
                 "n_updates":n_updates,
                 "cumulative_model_updates":self.cumulative_model_updates,
                 "mean_entropy":mean_entropy,
                 "mean_kl":mean_divergence,
                 "val_loss":mean_val_loss,
                 "clip_fraction":mean_clip,
                 "learning_rate":lr_report
                  }

        return report
