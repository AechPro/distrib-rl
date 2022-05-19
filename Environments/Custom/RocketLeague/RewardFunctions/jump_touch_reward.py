from rlgym.utils import RewardFunction
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.common_values import CEILING_Z

import numpy as np

class JumpTouchReward(RewardFunction):
    """
    a ball touch reward that only triggers when the agent's wheels aren't in contact with the floor
    adjust minimum ball height required for reward with 'min_height' as well as reward scaling with 'exp'
    """
    
    def __init__(self, min_height=92, exp=2):
        self.min_height = min_height
        self.exp = exp
        self.max_height = CEILING_Z - self.min_height
    
    def reset(self, initial_state: GameState):
        pass
    
    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            clipped_height = np.clip(state.ball.position[2] - self.min_height, 0, self.max_height)
    
            return np.power(clipped_height / self.max_height, self.exp)
    
        return 0
