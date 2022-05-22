from distrib_rl.Experience import Timestep, Trajectory
import numpy as np
import torch
import time

class BaseAgent(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.leftover_obs = None
        self.ep_rewards = []
        self.current_ep_rew = 0

    @torch.no_grad()
    def gather_timesteps(self, policy, env, num_timesteps=None, num_seconds=None, num_eps=None, trajectory_callback=None):
        trajectories = []
        trajectory = Trajectory()
        if self.leftover_obs is None:
            obs = env.reset()
        else:
            obs = self.leftover_obs

        cumulative_timesteps = 0
        start_time = time.time()
        while True:
            ts = Timestep()

            # ts.action and ts.log_prob will be filled here
            action = self._get_policy_action(policy, obs, ts)
            next_obs, rew, done, _ = env.step(action)

            self.current_ep_rew += rew
            ts.reward = rew
            ts.obs = obs
            ts.done = 1 if done else 0
            trajectory.register_timestep(ts)
            cumulative_timesteps += 1

            if done:
                self.ep_rewards.append(self.current_ep_rew)
                self.current_ep_rew = 0

                trajectory.final_obs = next_obs
                trajectories.append(trajectory)

                if trajectory_callback:
                    trajectory_callback(trajectory)
                trajectory = Trajectory()

                next_obs = env.reset()

            obs = next_obs
            if num_timesteps is not None and cumulative_timesteps >= num_timesteps or \
               num_seconds is not None and time.time() - start_time >= num_seconds or \
               num_eps is not None and len(trajectories) >= num_eps:
                break
            # print((time.perf_counter()-start_time)/cumulative_timesteps," | ",act_time/cumulative_timesteps," | ",step_time/cumulative_timesteps)
            # env.render()
        self.leftover_obs = next_obs

        if len(trajectory.obs) > 0:
            trajectory.final_obs = next_obs
            trajectories.append(trajectory)
            if trajectory_callback:
                trajectory_callback(trajectory)

        return trajectories

    @torch.no_grad()
    def evaluate_policy(self, policy, env, num_timesteps=0, num_eps=1, render=False):
        obs = env.reset()
        reward = 0
        rewards = []
        i = 0
        ts = Timestep()

        while i < num_timesteps or len(rewards) < num_eps:
            action = self._get_policy_action(policy, obs, ts, evaluate=False)
            next_obs, rew, done, _ = env.step(action)
            reward += rew

            if done:
                next_obs = env.reset()
                rewards.append(reward)
                reward = 0

            obs = next_obs
            i += 1

            if render:
                env.render()

        return np.mean(rewards)

    def _get_policy_action(self, policy, obs, timestep, evaluate=False):
        raise NotImplementedError
