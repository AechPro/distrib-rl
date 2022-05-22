import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward
from rlgym.utils import common_values

class SoloDefenderRewardFunction(RewardFunction):
    def __init__(self):
        self.y_threshold = -5120 + 900
        self.has_touched = False

    def reset(self, initial_state: GameState):
        self.has_touched = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        rew = 0
        if player.ball_touched:
            self.has_touched = True

        if not self.has_touched:
            bpos = state.ball.position
            ppos = player.car_data.position
            rew -= abs(bpos[0] - ppos[0] + state.ball.linear_velocity[0]) / (10*common_values.SIDE_WALL_X)

        if state.ball.position[1] < self.y_threshold:
            rew = -1

        return rew
