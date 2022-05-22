from rlgym.utils.terminal_conditions.common_conditions import BallTouchedCondition, GoalScoredCondition, NoTouchTimeoutCondition, TimeoutCondition
from rlgym_tools.extra_terminals.game_condition import GameCondition

from distrib_rl.Utils.FactoryBuilder import build_component_factory


_builders = {
    "ball_touched": BallTouchedCondition,
    "goal_scored": GoalScoredCondition,
    "no_touch_timeout": NoTouchTimeoutCondition,
    "timeout": TimeoutCondition,
    "game": GameCondition
}

_arg_transformers = {}

register_terminal_condition, build_terminal_conditions_from_config = build_component_factory(
    component_name = "terminal condition",
    builders = _builders,
    arg_transformers = _arg_transformers
)