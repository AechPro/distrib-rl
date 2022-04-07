from typing import Any

import gym.spaces
import numpy as np
from rlgym.utils.action_parsers import ActionParser, ContinuousAction
from rlgym.utils.gamestates import GameState


class SoloDefenderActionParser(ActionParser):
    def __init__(self):
        super().__init__()
        self.ap = ContinuousAction()

    def get_action_space(self) -> gym.spaces.Space:
        return self.ap.get_action_space()

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self.ap.parse_actions(actions, state)
