import stable_baselines3
import numpy as np
from Environments.Custom import MinAtarWrapper
import gym

def match_fn():
    from Environments.Custom.rlgym.envs import Match
    from Environments.Custom.rlgym.utils.reward_functions import DefaultReward
    from Environments.Custom.rlgym.utils.obs_builders import AdvancedObs
    from Environments.Custom.rlgym.utils.state_setters import DefaultState
    from Environments.Custom.rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, \
        GoalScoredCondition, NoTouchTimeoutCondition

    ep_len_min = 3
    min_per_touch = 20 / 60
    tick_skip = 8
    ticks_per_touch = min_to_ticks(min_per_touch, tick_skip)
    ep_len_ticks = min_to_ticks(ep_len_min, tick_skip)

    conditions = [NoTouchTimeoutCondition(ticks_per_touch), TimeoutCondition(ep_len_ticks), GoalScoredCondition()]
    obs_builder = AdvancedObs()

    reward_fn = DefaultReward()

    match = Match(reward_fn, conditions, obs_builder, DefaultState(),
                  team_size=1, tick_skip=tick_skip, game_speed=100,
                  spawn_opponents=True, self_play=True)

    return match

def min_to_ticks(ep_len_min, tick_skip):
    ticks_per_min = 120 * 60
    return int(round(ep_len_min * ticks_per_min / tick_skip))

def make_rlgym():
    from PolicyOptimization.SB3.SB3RLGymStuff import SB3MultipleInstanceEnv
    from stable_baselines3.common.vec_env import VecMonitor


    wrapper = VecMonitor(SB3MultipleInstanceEnv("H:\\EpicLibrary\\rocketleague\\Binaries\\Win64\\RocketLeague.exe",
                                     match_fn, num_instances=2, wait_time=15, force_paging=True))


    return wrapper


class SB3Optimizer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        #self.env = MinAtarWrapper(cfg["env_id"])
        self.env = gym.make(cfg["env_id"])
        self.env.seed(cfg["seed"])
        self.env.action_space.seed(cfg["seed"])
        #self.env = make_rlgym()

    def train(self):
        n_envs = 1
        max_kl = 100

        n_epochs = self.cfg["policy_optimizer"]["n_epochs"]
        batch_size = self.cfg["policy_optimizer"]["batch_size"]
        n_steps = self.cfg["policy_optimizer"]["timesteps_per_update"] // n_envs
        clip_range = self.cfg["policy_optimizer"]["clip_range"]
        gamma = self.cfg["policy_optimizer"]["gamma"]
        gae_lambda = self.cfg["policy_optimizer"]["gae_lambda"]
        learning_rate = self.cfg["policy_gradient_optimizer"]["lr"]
        max_ts = int(self.cfg["policy_optimizer"]["max_timesteps"])

        """
        "type": "pg",
        "max_timesteps": 1e9,
        "n_epochs": 10,
        "batch_size": 5000,
        "timesteps_per_update": 20000,
        "value_updates_per_batch": 10,
        "eps_per_eval": 5,
        "clip_fraction": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "max_kl": 0.06,
        "entropy_coef": 0.02"""

        learner = stable_baselines3.PPO("MlpPolicy", self.env, verbose=1,
                                        n_epochs=n_epochs,
                                        n_steps=n_steps,
                                        batch_size=batch_size,
                                        clip_range=clip_range,
                                        gamma=gamma,
                                        gae_lambda=gae_lambda,
                                        target_kl=max_kl,
                                        vf_coef=1,
                                        learning_rate=learning_rate,
                                        )
        learner.learn(max_ts)
