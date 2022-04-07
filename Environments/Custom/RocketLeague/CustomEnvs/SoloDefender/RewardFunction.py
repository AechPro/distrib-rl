import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
from rlgym.utils import common_values

class SoloDefenderRewardFunction(RewardFunction):
    def __init__(self):
        self.y_threshold = -5120 + 900
        self.vel_rew = VelocityPlayerToBallReward()

    def reset(self, initial_state: GameState):
        self.vel_rew.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        rew += self.vel_rew.get_reward(player, state, previous_action) / common_values.CAR_MAX_SPEED
        if state.ball.position[1] < self.y_threshold:
            rew = -1
        return rew
