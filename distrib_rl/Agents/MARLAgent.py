from distrib_rl.Experience import Timestep, Trajectory
from distrib_rl.Policies import PolicyFactory
from distrib_rl.MARL import OpponentSelector
import numpy as np
import torch
import time

class MARLAgent(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.leftover_obs = None
        self.opponent_policy = None
        self.opponent_num = -1
        self.opponent_selector = OpponentSelector(cfg)
        self.save_both_teams = True
        self.ep_rewards = []
        self.current_ep_rew = 0
        self.policies = None
        self.n_agents = cfg["rlgym"]["team_size"] * 2 if cfg["rlgym"]["spawn_opponents"] else cfg["rlgym"]["team_size"]


    @torch.no_grad()
    def gather_timesteps(self, policy, env, num_timesteps=None, num_seconds=None, num_eps=None, trajectory_callback=None):
        n_agents = self.n_agents
        agents_to_save = n_agents if self.save_both_teams else n_agents // 2

        if self.opponent_policy is None:
            self.init_opponent_policy(env)

        if self.policies is None:
            self.policies = [policy for _ in range(n_agents // 2)] + [self.opponent_policy for _ in range(n_agents // 2)]
        policies = self.policies
        experience_trajectories = []
        trajectories = [Trajectory() for _  in range(n_agents)]

        obs = self.leftover_obs
        if obs is None:
            obs = env.reset()

        cumulative_timesteps = 0
        start_time = time.time()
        while True:
            actions = []
            ts = [Timestep() for _ in range(n_agents)]
            for i in range(n_agents):
                action, log_prob = policies[i].get_action(obs[i], deterministic=False)
                ts[i].action = action
                ts[i].log_prob = log_prob
                actions.append(action)

            next_obs, rews, done, _ = env.step(np.asarray(actions))

            for i in range(n_agents):
                if self.save_both_teams:
                    self.current_ep_rew += rews[i]
                elif i < n_agents//2:
                    self.current_ep_rew += rews[i]

                ts[i].reward = rews[i]
                ts[i].obs = obs[i].copy()
                ts[i].done = 1 if done else 0
                trajectories[i].register_timestep(ts[i])

            cumulative_timesteps += 1
            if done:

                self.ep_rewards.append(self.current_ep_rew/agents_to_save)
                self.current_ep_rew = 0

                for i in range(agents_to_save):
                    trajectories[i].final_obs = next_obs[i]
                    experience_trajectories.append(trajectories[i])
                    if trajectory_callback:
                        trajectory_callback(trajectories[i])

                # todo: Implement a proper opponent evaluation & selection scheme and delete this.
                result = sum(trajectories[0].rewards) > sum(trajectories[-1].rewards)
                self.opponent_selector.submit_result(self.opponent_num, result)
                self.get_next_opponent(policy)

                next_obs = env.reset()
                trajectories = [Trajectory() for _  in range(n_agents)]

            obs = next_obs
            if num_timesteps is not None and cumulative_timesteps >= num_timesteps or \
               num_seconds is not None and time.time() - start_time >= num_seconds or \
               num_eps is not None and len(experience_trajectories) >= num_eps:
                break

        self.leftover_obs = next_obs.copy()

        for i in range(agents_to_save):
            trajectories[i].final_obs = next_obs[i]
            experience_trajectories.append(trajectories[i])
            if trajectory_callback:
                trajectory_callback(trajectories[i])

        return experience_trajectories

    def get_next_opponent(self, policy):
        opponent_weights, opponent_num = self.opponent_selector.get_opponent()
        if type(opponent_weights) not in (tuple, list, np.ndarray, np.array):
            self.opponent_policy.set_trainable_flat(policy.get_trainable_flat().copy())
        else:
            self.opponent_policy.set_trainable_flat(opponent_weights.copy())
        self.opponent_num = opponent_num
        self.save_both_teams = opponent_num == -1

    def init_opponent_policy(self, env):
        models = PolicyFactory.get_from_cfg(self.cfg, env)
        self.opponent_policy = models["policy"]
        self.opponent_policy.to(self.cfg["device"])
        models.clear()

    def _get_policy_action(self, policy, obs, timestep, evaluate=False):
        raise NotImplementedError
