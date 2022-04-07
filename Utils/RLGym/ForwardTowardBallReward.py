import numpy as np
from Environments.Custom.rlgym.utils.gamestates import PlayerData, GameState
from Environments.Custom.rlgym.utils.reward_functions import RewardFunction
from Environments.Custom.rlgym.utils import math, common_values


class ForwardTowardBallReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        vel = player.car_data.linear_velocity
        forward_vel = math.scalar_projection(vel, player.car_data.forward()) / common_values.CAR_MAX_SPEED
        pos_diff = state.ball.position - player.car_data.position
        vel_to_ball = math.scalar_projection(vel, pos_diff) / common_values.CAR_MAX_SPEED

        if vel_to_ball < 0 and forward_vel < 0:
            rew = vel_to_ball*abs(forward_vel)
        else:
            rew = vel_to_ball * forward_vel

        return rew