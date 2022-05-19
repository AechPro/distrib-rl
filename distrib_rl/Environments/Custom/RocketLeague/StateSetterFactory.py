from rlgym.utils.state_setters import DefaultState, RandomState
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter

from distrib_rl.Utils.FactoryBuilder import build_component_factory

_builders = {
    "default": DefaultState,
    "kickoff": DefaultState,
    "random": RandomState,
    "augment_setter": AugmentSetter,
    "goalie": GoaliePracticeState,
    "hoops": HoopsLikeSetter,
    "replay": ReplaySetter,
    "wall": WallPracticeState,
    "kickoff_like": KickoffLikeSetter,
    "weighted_sample": WeightedSampleSetter
}

_arg_transformers = {
    "augment_setter": lambda **kwargs: {
        "state_setter": build_state_setter_from_config(kwargs["state_setter"]),
        "shuffle_within_teams": kwargs.get("shuffle_within_teams", True),
        "swap_front_back": kwargs.get("swap_front_back", True)
    },
    "weighted_sample": lambda **kwargs: {
        "state_setters": [build_state_setter_from_config(s) for s in kwargs["state_setters"]],
        "weights": kwargs["weights"]
    }
}

register_state_setter, build_state_setter_from_config = build_component_factory(
    component_name ="state setter",
    builders = _builders,
    arg_transformers=_arg_transformers
)