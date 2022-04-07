from Experience import Trajectory, Timestep
import torch
import numpy as np

from Utils import WelfordRunningStat


class ExperienceReplay(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.actions        = torch.FloatTensor()
        self.log_probs      = torch.FloatTensor()
        self.rewards        = torch.FloatTensor()
        self.obs            = torch.FloatTensor()
        self.dones          = torch.FloatTensor()
        self.future_rewards = torch.FloatTensor()
        self.values         = torch.FloatTensor()
        self.advantages     = torch.FloatTensor()
        self.pred_rets      = torch.FloatTensor()
        self.ep_rews        = torch.FloatTensor()
        self.noise_idxs     = torch.FloatTensor()
        self.num_timesteps = 0
        self.max_buffer_size = cfg["experience_replay"]["max_buffer_size"]
        self.rng = cfg["rng"]

        self.reward_stats = WelfordRunningStat(1)

        self.time = 0

    def register_trajectory(self, trajectory : Trajectory, serialized=False):
        if not serialized:
            actions, log_probs, rewards, obs, dones, future_rewards, values, advantages, pred_rets, ep_rew, noise_idx = trajectory.serialize()
        else:
            actions, log_probs, rewards, obs, dones, future_rewards, values, advantages, pred_rets, ep_rew, noise_idx = trajectory

        self.reward_stats.increment(future_rewards, len(future_rewards))
        self.actions = torch.cat((self.actions, torch.as_tensor(actions, dtype=torch.float32)), 0)
        self.log_probs = torch.cat((self.log_probs, torch.as_tensor(log_probs, dtype=torch.float32)), 0)
        self.obs = torch.cat((self.obs, torch.as_tensor(obs, dtype=torch.float32)), 0)
        self.values = torch.cat((self.values, torch.as_tensor(values, dtype=torch.float32)), 0)
        self.advantages = torch.cat((self.advantages, torch.as_tensor(advantages, dtype=torch.float32)), 0)
        self._clamp_size()
        self.num_timesteps = len(self.actions)


    def get_all_batches_shuffled(self, batch_size):
        if batch_size == self.num_timesteps:
            return self.get_all_batches(batch_size)

        indices = [i for i in range(self.num_timesteps)]
        self.rng.shuffle(indices)

        acts, probs, rews, obs, dones, f_rews, vals, adv, pr = self.actions, self.log_probs, self.rewards, self.obs, \
                                                                      self.dones,  self.future_rewards, \
                                                                      self.values, self.advantages, self.pred_rets

        acts = acts[indices]
        probs = probs[indices]
        obs = obs[indices]
        vals = vals[indices]
        adv = adv[indices]

        batches = []
        n = len(acts) // batch_size

        # max_idx = self.num_timesteps - batch_size
        # indices = self.rng.randint(0, max_idx, n)
        #self.cfg["rng"].shuffle(indices)

        for i in range(n):
            # batch = Trajectory()
            start = i * batch_size
            stop = start + batch_size

            # batch.actions = acts[start:stop]
            # batch.log_probs = probs[start:stop]
            # batch.obs = obs[start:stop]
            # batch.values = vals[start:stop]
            # batch.advantages = adv[start:stop]

            #batches.append(batch)

            batches.append([
                acts[start:stop],
                probs[start:stop],
                obs[start:stop],
                vals[start:stop],
                adv[start:stop]
            ])


        return batches

    def get_all_batches(self, batch_size):
        acts, probs, rews, obs, dones, f_rews, vals, adv, pr = self.actions, self.log_probs, self.rewards, self.obs, \
                                                                  self.dones, self.future_rewards, \
                                                                  self.values, self.advantages, self.pred_rets

        batches = []
        n = len(acts) // batch_size

        for i in range(n):
            batch = Trajectory()
            start = i*batch_size
            stop = start + batch_size

            batch.actions = acts[start:stop]
            batch.log_probs = probs[start:stop]
            batch.obs = obs[start:stop]
            batch.values = vals[start:stop]
            batch.advantages = adv[start:stop]

            batches.append(batch)

        return batches

    def get_all(self):
        return self.actions, self.log_probs, self.rewards, self.obs, self.dones, self.future_rewards,\
               self.values, self.advantages, self.pred_rets

    def get_batch(self, size):
        acts, probs, rews, obs, dones, f_rews, vals, adv, pr = self.actions, self.log_probs, self.rewards, self.obs,\
                                                                  self.dones, self.future_rewards, \
                                                                  self.values, self.advantages, self.pred_rets

        if size > self.num_timesteps:
            print("Asked for batch of size {} when only {} timesteps have been collected. Returning entire memory.".
                  format(size, self.num_timesteps))

            return acts, probs, rews, obs, dones, f_rews, vals, adv, pr

        return acts[:size], probs[:size], rews[:size], obs[:size], dones[:size], f_rews[:size], \
               vals[:size], adv[:size], pr[:size]

    def get_random_batch(self, size):
        acts, probs, rews, obs, dones, f_rews, vals, adv, pr = self.actions, self.log_probs, self.rewards, self.obs, \
                                                                  self.dones, self.future_rewards, \
                                                                  self.values, self.advantages, self.pred_rets
        size = min(size, self.num_timesteps)
        indices = [i for i in range(self.num_timesteps)]
        self.cfg["rng"].shuffle(indices)
        indices = indices[:size]

        pr = pr[indices]
        acts = acts[indices]
        probs = probs[indices]
        rews = rews[indices]
        obs = obs[indices]
        dones = dones[indices]
        f_rews = f_rews[indices]
        vals = vals[indices]
        adv = adv[indices]

        return acts, probs, rews, obs, dones, f_rews, vals, adv, pr


    def _clamp_size(self):
        arr_to_check = max(len(self.actions), len(self.noise_idxs))
        start = arr_to_check - self.max_buffer_size
        if start > 0:
            self.actions = self.actions[start:]
            self.log_probs = self.log_probs[start:]
            self.rewards = self.rewards[start:]
            self.obs = self.obs[start:]
            self.dones = self.dones[start:]
            self.future_rewards = self.future_rewards[start:]
            self.values = self.values[start:]
            self.advantages = self.advantages[start:]
            self.pred_rets = self.pred_rets[start:]
            self.noise_idxs = self.noise_idxs[start:]
            self.ep_rews = self.ep_rews[start:]

    def clear(self):
        self.__init__(self.cfg)