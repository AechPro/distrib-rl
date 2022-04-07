import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward


class DefaultRewardFunction(RewardFunction):
    def __init__(self):
        self.rf = CombinedReward(
            (VelocityPlayerToBallReward(),
             VelocityBallToGoalReward(),
             EventReward(goal=1, concede=-1),
            ),
            (0.1,2,10)
        )

    def reset(self, initial_state: GameState):
        self.rf.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.rf.get_reward(player, state, previous_action)
