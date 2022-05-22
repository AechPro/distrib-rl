from rlgym.utils.action_parsers import ContinuousAction, DefaultAction, DiscreteAction
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from .ActionParsers import NectoActionParser

from distrib_rl.Utils.FactoryBuilder import build_component_factory

_builders = {
    "continuous": ContinuousAction,
    "default": DefaultAction,
    "discrete": DiscreteAction,
    "kbm": KBMAction,
    "necto": NectoActionParser
}

_arg_transformers = {}

register_action_parser, build_action_parser_from_config = build_component_factory(
    component_name="action parser",
    builders=_builders,
    arg_transformers=_arg_transformers
)
