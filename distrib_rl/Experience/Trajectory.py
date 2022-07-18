from distrib_rl.Experience import Timestep
from distrib_rl.Utils import MathHelpers as RLMath
import numpy as np
import torch

class Trajectory(object):
    def __init__(self):
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.obs = []
        self.dones = []
        self.next_obs = []
        self.future_rewards = []
        self.values = []
        self.advantages = []
        self.pred_rets = []
        self.final_obs = None
        self.is_partial = False
        self.ep_rew = 0
        self.noise_idx = 0

    def register_timestep(self, timestep : Timestep):
        action, log_prob, reward, obs, done = timestep.serialize()
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.obs.append(obs)
        self.dones.append(done)

    def serialize(self):
        return (self.actions, self.log_probs, self.rewards, self.obs, self.dones,
                      self.future_rewards, self.values, self.advantages, self.pred_rets, self.ep_rew, self.noise_idx)

    def deserialize(self, other):
        self.actions, self.log_probs, self.rewards, self.obs, self.dones, \
        self.future_rewards, self.values, self.advantages, self.pred_rets, self.ep_rew, self.noise_idx = other


    def truncate(self, stop):
        self.actions = self.actions[:stop]
        self.log_probs = self.log_probs[:stop]
        self.rewards = self.rewards[:stop]
        self.obs = self.obs[:stop]
        self.dones = self.dones[:stop]
        self.future_rewards = self.future_rewards[:stop]
        self.values = self.values[:stop]
        self.advantages = self.advantages[:stop]
        self.pred_rets = self.pred_rets[:stop]

    @torch.no_grad()
    def finalize(self, gamma=None, lmbda=None, values=None, reward_stats=None):
        if reward_stats is not None:
            mean, std = reward_stats
            # rews = np.divide(np.subtract(self.rewards, mean), std)
            # rews = np.divide(self.rewards, std)
            # self.rewards = rews.tolist()

        if gamma is not None:
            self.future_rewards = RLMath.compute_discounted_future_sum(self.rewards, gamma).tolist()

        if values is not None:
            self.values = [arg for arg in values]

        if lmbda is not None:
            if reward_stats is not None:
                mean, std = reward_stats
                rews = np.divide(self.rewards, std)
            else:
                rews = self.rewards

            next_values = values[1:]
            terminal = self.dones

            last_gae_lam = 0
            n_returns = len(rews)
            adv = [0 for _ in range(n_returns)]
            self.pred_rets = [0 for _ in range(n_returns)]

            for step in reversed(range(n_returns)):
                if step == n_returns - 1:
                    done = 1 - terminal[-1]
                else:
                    done = 1 - terminal[step + 1]

                pred_ret = rews[step] + gamma * next_values[step] * done
                self.pred_rets[step] = pred_ret
                delta = pred_ret - values[step]
                last_gae_lam = delta + gamma * lmbda * done * last_gae_lam
                adv[step] = last_gae_lam

            self.advantages = adv
            self.values = [v + a for v, a in zip(values[:-1], adv)]