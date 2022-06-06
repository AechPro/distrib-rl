from distrib_rl.Distrib import RedisKeys
from distrib_rl.Experience import ExperienceReplay

class DistribExperienceManager(object):
    def __init__(self, cfg, client=None, server=None):
        self.cfg = cfg
        self.client = client
        self.server = server
        self.experience = ExperienceReplay(cfg)

    def push_trajectories(self, trajectories):
        if self.client is None:
            return

        self.client.push_data(RedisKeys.CLIENT_EXPERIENCE_KEY, trajectories)

    def get_timesteps_as_batches(self, num_timesteps, batch_size):
        if self.server is None:
            return None

        exp = self.experience

        n_collected = 0
        acts = []
        probs = []
        rews = []
        obses = []
        done = []
        frews = []
        vals = []
        advs = []
        rets = []
        noise_indices = []
        ep_rews = []

        while True:
            batches = self.server.get_n_timesteps(num_timesteps)
            for batch in batches:
                actions, log_probs, rewards, obs, dones, future_rewards, values, advantages, pred_rets, ep_rew, noise_idx = batch
                acts += actions
                probs += log_probs
                rews += rewards
                obses += obs
                done += dones
                frews += future_rewards
                vals += values
                advs += advantages
                rets += pred_rets
                noise_indices.append(noise_idx)
                ep_rews.append(ep_rew)
                n_collected += len(batch[0])

            if len(acts) > 0:
                exp.register_trajectory((acts, probs, rews, obses, done, frews, vals, advs, rets, noise_indices, ep_rews), serialized=True)
                break

            if exp.num_timesteps > batch_size:
                break

        return n_collected, self.server.steps_per_second

    def cleanup(self):
        self.experience.clear()
