import stable_baselines3
import gym

def match_fn():
    from rlgym.envs import Match
    from rlgym.utils.reward_functions import DefaultReward
    from rlgym.utils.obs_builders import AdvancedObs
    from rlgym.utils.state_setters import DefaultState
    from rlgym.utils.action_parsers import DiscreteAction
    from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, \
        GoalScoredCondition, NoTouchTimeoutCondition

    ep_len_min = 3
    min_per_touch = 20 / 60
    tick_skip = 8
    ticks_per_touch = min_to_ticks(min_per_touch, tick_skip)
    ep_len_ticks = min_to_ticks(ep_len_min, tick_skip)

    conditions = [NoTouchTimeoutCondition(ticks_per_touch), TimeoutCondition(ep_len_ticks), GoalScoredCondition()]
    obs_builder = AdvancedObs()

    reward_fn = DefaultReward()

    match = Match(reward_fn,
                 conditions,
                 obs_builder,
                 DiscreteAction(),
                 DefaultState(),
                  team_size=1, tick_skip=tick_skip, game_speed=100, spawn_opponents=True, self_play=True)

    return match

def min_to_ticks(ep_len_min, tick_skip):
    ticks_per_min = 120 * 60
    return int(round(ep_len_min * ticks_per_min / tick_skip))

def make_rlgym():
    from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
    from stable_baselines3.common.vec_env import VecMonitor


    wrapper = VecMonitor(SB3MultipleInstanceEnv(match_fn, num_instances=1, wait_time=30, force_paging=False))


    return wrapper


class SB3Optimizer(object):
    def __init__(self):
        self.env = make_rlgym()

    def train(self):
        n_envs = 1
        n_agents = 2
        max_kl = 100

        n_epochs = 10
        batch_size = 32
        n_steps = 2048 // (n_envs*n_agents)
        clip_range = 0.2
        gamma = 0.99
        gae_lambda = 0.97
        learning_rate = 3e-4
        max_ts = 100_000_000

        learner = stable_baselines3.PPO("MlpPolicy", self.env, verbose=1,
                                        n_epochs=n_epochs,
                                        n_steps=n_steps,
                                        batch_size=batch_size,
                                        clip_range=clip_range,
                                        gamma=gamma,
                                        gae_lambda=gae_lambda,
                                        target_kl=max_kl,
                                        vf_coef=0.5,
                                        learning_rate=learning_rate,
                                        )
        learner.learn(max_ts)

if __name__ == "__main__":
    opt = SB3Optimizer()
    opt.train()