from rlgym.utils.gamestates import GameState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.terminal_conditions import TerminalCondition

class SoloDefenderTerminalConditions(TerminalCondition):
    def __init__(self):
        super().__init__()
        self.conditions = [TimeoutCondition(100), GoalScoredCondition()]

    def reset(self, initial_state: GameState):
        for condition in self.conditions:
            condition.reset(initial_state)

    def is_terminal(self, current_state: GameState) -> bool:
        if current_state.ball.position[1] < -5120 + 900:
            return True

        for condition in self.conditions:
            if condition.is_terminal(current_state):
                return True
        return False
