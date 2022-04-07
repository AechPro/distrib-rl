from typing import Any

import numpy as np
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder, DefaultObs
from rlgym.utils import common_values


class SoloDefenderObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.ob = DefaultObs(pos_coef=np.asarray([1/common_values.SIDE_WALL_X, 1/common_values.BACK_WALL_Y, 1/common_values.CEILING_Z]),
                             ang_coef=1/np.pi,
                             lin_vel_coef=1/common_values.CAR_MAX_SPEED,
                             ang_vel_coef=1/common_values.CAR_MAX_ANG_VEL)

    def reset(self, initial_state: GameState):
        self.ob.reset(initial_state)

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        return self.ob.build_obs(player, state, previous_action)
