import numpy as np
from Environments.Custom.rlgym.utils.gamestates import PlayerData, GameState
from Environments.Custom.rlgym.utils.reward_functions import RewardFunction
from Environments.Custom.rlgym.utils import math, common_values


class ForwardReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        forward_vel = math.scalar_projection(vel, player.car_data.forward()) / common_values.CAR_MAX_SPEED

        return forward_vel