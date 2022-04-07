import redis
import numpy as np
from Distrib import RedisKeys
from Environments.Custom.rlgym.utils.reward_functions import RewardFunction
from Environments.Custom.rlgym.utils.gamestates import PlayerData, GameState
from Environments.Custom.rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward, EventReward, VelocityPlayerToBallReward
from Utils.RLGym import ForwardTowardBallReward, ForwardReward


class RedisAnnealReward(RewardFunction):
    def __init__(self):
        super().__init__()

        self.rewards = [VelocityPlayerToBallReward(), VelocityBallToGoalReward(), EventReward(goal=1, concede=-1)]
        self.reward_thresholds = [180, 100]
        self.anneal_times = [50_000_000, 200_000_000]

        self.current_anneal_time = None
        self.current_threshold = None
        self.current_reward = None
        self.next_reward = None

        self.is_annealing = False
        self.annealing_ts = 0

        self.redis = redis.Redis(port=25565)

        self._advance()

    def reset(self, initial_state: GameState):
        self.current_reward.reset(initial_state)

        if self.next_reward is not None:
            self.next_reward.reset(initial_state)

            if not self.is_annealing and self.current_threshold is not None:
                rew = self.redis.get(RedisKeys.MEAN_POLICY_REWARD_KEY)

                if rew is None:
                    policy_reward = -np.inf
                else:
                    policy_reward = float(rew)

                if policy_reward >= self.current_threshold:
                    self.is_annealing = True
                    self.annealing_ts = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if self.is_annealing:
            self.annealing_ts += 1

            frac = self.annealing_ts / self.current_anneal_time
            r1 = self.current_reward.get_reward(player, state, previous_action)
            r2 = self.next_reward.get_reward(player, state, previous_action)
            rew = r1 * (1 - frac) + r2 * frac

            if self.annealing_ts >= self.current_anneal_time:
                self._advance()

        else:
            rew = self.current_reward.get_reward(player, state, previous_action)

        return rew

    def _advance(self):
        self.is_annealing = False
        self.annealing_ts = 0

        if len(self.rewards) == 0:
            return

        self.current_reward = self.rewards.pop(0)

        if len(self.rewards) == 0:
            self.next_reward = None
            self.current_threshold = None
            self.current_anneal_time = None
        else:
            self.next_reward = self.rewards[0]
            self.current_threshold = self.reward_thresholds.pop(0)
            self.current_anneal_time = self.anneal_times.pop(0)