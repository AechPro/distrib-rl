from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, BallTouchedCondition
from rlgym.utils.terminal_conditions import TerminalCondition

class KickoffTerminalConditions(TerminalCondition):
    def __init__(self):
        super().__init__()
        self.conditions = [TimeoutCondition(80), BallTouchedCondition()]

    def reset(self, initial_state: GameState):
        for condition in self.conditions:
            condition.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        for condition in self.conditions:
            if condition.is_terminal(current_state):
                return True
        return False
