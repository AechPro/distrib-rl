from distrib_rl.PolicyOptimization.PolicyGradients import Configurator
from distrib_rl.Utils import MathHelpers as RLMath
import torch
import torch.nn.functional as fn
import numpy as np
import time


class PolicyGradients(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.env, \
        self.experience, \
        self.gradient_builder, \
        self.policy_gradient_optimizer, \
        self.value_gradient_optimizer, \
        self.agent, \
        self.policy, \
        self.strategy_optimizer, \
        self.adaptive_omega, \
        self.value_net, \
        self.novelty_gradient_optimizer, \
        self.learner = Configurator.build_vars(cfg)

        self.prev_mean = 0
        self.value_loss_fn = torch.nn.MSELoss()
        self.epoch = 0
        self.epoch_info = {}

    def train(self):
        ts_consumed = 0
        self.epoch = 0
        while ts_consumed < int(self.cfg["policy_optimizer"]["max_timesteps"]):
            t1 = time.time()
            self.experience.clear()
            self.collect()

            before = self.policy.get_trainable_flat()
            self.update_models()
            after = self.policy.get_trainable_flat(force_update=True)
            self.epoch_info["update_magnitude"] = np.linalg.norm(np.subtract(before, after))

            #self.strategy_optimizer.update()
            #rew = self.agent.evaluate_policy(self.policy, self.env, num_eps=25)

            rews = self.agent.ep_rewards
            rew = 0 if len(rews) < 10 else np.mean(rews[:-10])

            self.adaptive_omega.step(rew)

            #self.epoch_info["omega"] = self.adaptive_omega.omega
            self.epoch_info["mean_policy_reward"] = rew
            #self.epoch_info["policy_novelty"] = self.strategy_optimizer.compute_policy_novelty()
            self.epoch_info["epoch"] = self.epoch
            self.epoch_info["epoch_time"] = time.time() - t1
            ts_consumed += self.epoch_info["ts_consumed"]

            self.report_epoch()
            self.epoch_info.clear()
            self.epoch += 1

    @torch.no_grad()
    def collect(self):
        trajectories = list(self.agent.gather_timesteps(self.policy, self.epoch, self.env,
                                                        num_timesteps=self.cfg["policy_optimizer"]["timesteps_per_update"]))
        value_estimator = self.value_net

        rewards = []
        for trajectory in trajectories:
            rewards += trajectory.rewards

        mean = np.mean(rewards)
        std = np.std(rewards)
        if std == 0:
            std = 1

        self.epoch_info["mean_ep_rew"] = mean

        for trajectory in trajectories:
            values = value_estimator.get_output(trajectory.obs).flatten().numpy().tolist()
            final_val = value_estimator.get_output(trajectory.next_obs[-1]).numpy().tolist()
            values.append(final_val[0])
            trajectory.finalize(gamma=self.cfg["policy_optimizer"]["gamma"], reward_stats=None, values=values,
                                lmbda=self.cfg["policy_optimizer"]["gae_lambda"])
            self.experience.register_trajectory(trajectory)

    def update_models(self):
        batch_size = self.cfg["policy_optimizer"]["batch_size"]
        #value_updates_per_batch = self.cfg["policy_optimizer"]["value_updates_per_batch"]

        batches = self.experience.get_all_batches(batch_size)
        num_batches = len(batches)
        self.epoch_info["learner"] = self.learner.learn(batches)
        self.update_policy_novelty()

        self.epoch_info["redis_batches"] = num_batches
        self.epoch_info["ts_consumed"] = num_batches * self.cfg["policy_optimizer"]["batch_size"]

    def update_policy_novelty(self):
        w = self.adaptive_omega.omega

        # gradient = -self.strategy_optimizer.get_novelty_gradient()
        #
        # self.gradient_builder.contribute_gradient_from_flat(gradient, w)
        # self.gradient_builder.update_model(self.policy, self.novelty_gradient_optimizer)

        self.epoch_info["policy_novelty"] = self.strategy_optimizer.compute_policy_novelty()
        self.epoch_info["omega"] = w


    def report_epoch(self):
        info = self.epoch_info

        asterisks = "*" * 8
        report = "\n{} BEGIN EPOCH {} REPORT {}\n" \
                 "Policy Reward:         {:7.5f}\n" \
                 "Policy Novelty:        {:7.5f}\n" \
                 "Policy Entropy:        {:7.5f}\n" \
                 "Policy Updates:        {:7}\n\n" \
                 "KL Divergence:         {:7.5f}\n" \
                 "Clip Fraction:         {:7.5f}\n" \
                 "Update Magnitude:      {:7.5f}\n" \
                 "Omega:                 {:7.5f}\n\n" \
                 "TS This Epoch          {:7}\n" \
                 "Redis Batches          {:7}\n" \
                 "Value loss:            {:7.5f}\n\n" \
                 "Epoch Time:            {:7.5f}\n" \
                 "{} END EPOCH {} REPORT {}\n".format(
            asterisks,
            info["epoch"],
            asterisks,
            info["mean_policy_reward"],
            info["policy_novelty"],
            info["learner"]["mean_entropy"],
            info["learner"]["n_updates"],
            info["learner"]["mean_kl"],
            info["learner"]["clip_fraction"],
            info["update_magnitude"],
            info["omega"],
            info["ts_consumed"],
            info["redis_batches"],
            info["learner"]["val_loss"],
            info["epoch_time"],
            asterisks,
            info["epoch"],
            asterisks
        )
        print(report)
