from Experience import Timestep, Trajectory
from Policies import PolicyFactory
from MARL import OpponentSelector
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


    @torch.no_grad()
    def gather_timesteps(self, policy, env, num_timesteps=None, num_seconds=None, num_eps=None):
        if self.opponent_policy is None:
            self.init_opponent_policy(env)

        n_agents = 2
        policies = [policy, self.opponent_policy]
        experience_trajectories = []
        trajectories = [Trajectory() for _  in range(n_agents)]

        if self.leftover_obs is None:
            obs = env.reset()
        else:
            obs = self.leftover_obs

        cumulative_timesteps = 0
        start_time = time.time()
        while True:
            actions = []
            ts = [Timestep() for _ in range(n_agents)]
            for i in range(n_agents):
                action, log_prob = policies[i].get_action(obs[i])
                ts[i].action = action
                ts[i].log_prob = log_prob
                actions.append(action)

            next_obs, rews, done, _ = env.step(np.asarray(actions))

            for i in range(n_agents):
                if not self.save_both_teams and i < n_agents//2:
                    self.current_ep_rew += rews[i]
                elif self.save_both_teams:
                    self.current_ep_rew += rews[i]

                ts[i].reward = rews[i]
                ts[i].obs = obs[i].copy()
                ts[i].done = 1 if done else 0
                trajectories[i].register_timestep(ts[i])

            cumulative_timesteps += 1
            if done:
                num = n_agents if self.save_both_teams else n_agents // 2
                self.ep_rewards.append(self.current_ep_rew/num)
                self.current_ep_rew = 0

                for i in range(num):
                    trajectories[i].final_obs = next_obs[i]
                    experience_trajectories.append(trajectories[i])

                result = sum(trajectories[0].rewards) > sum(trajectories[1].rewards)
                self.opponent_selector.submit_result(self.opponent_num, result)
                self.get_next_opponent(policy)
                next_obs = env.reset()

                trajectories = [Trajectory() for _  in range(n_agents)]

            obs = next_obs
            if num_timesteps is not None and cumulative_timesteps >= num_timesteps or \
               num_seconds is not None and time.time() - start_time >= num_seconds or \
               num_eps is not None and len(experience_trajectories) >= num_eps:
                break

        if not done:
            self.leftover_obs = next_obs.copy()
        else:
            self.leftover_obs = None


        if len(trajectories[0].obs) > 0:
            trajectories[0].final_obs = next_obs[0]
            experience_trajectories.append(trajectories[0])

        if self.save_both_teams:
            for i in range(1, n_agents):
                if len(trajectories[i].obs) > 0:
                    trajectories[i].final_obs = next_obs[i]
                    experience_trajectories.append(trajectories[i])

        return experience_trajectories

    @torch.no_grad()
    def evaluate_policy(self, policy, env, num_timesteps=0, num_eps=1, render=False):
        obs = env.reset()
        reward1, reward2 = 0, 0
        rewards1, rewards2 = [], []
        i = 0

        if self.opponent_policy is None:
            self.init_opponent_policy(env)

        p1 = policy
        p2 = self.opponent_policy

        while i < num_timesteps or len(rewards1) < num_eps:
            obs1, obs2 = obs
            action1, log_prob1 = p1.get_action(obs1)
            action2, log_prob2 = p2.get_action(obs2)
            next_obs, rew, done, _ = env.step((action1, action2))

            reward1 += rew[0]
            reward2 += rew[1]

            if done:
                next_obs = env.reset()
                rewards1.append(reward1)
                rewards2.append(reward2)
                reward1 = 0
                reward2 = 0

            obs = next_obs
            i += 1

            if render:
                env.render()

        return np.mean(rewards1)

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
