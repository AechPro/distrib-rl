from rlgym.utils.obs_builders import AdvancedObs, DefaultObs
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym_tools.extra_obs.general_stacking import GeneralStacker

from Utils.FactoryBuilder import build_component_factory

_builders = {
    "default": DefaultObs,
    "advanced": AdvancedObs,
    "advanced_stacker": AdvancedStacker,
    "advanced_padder": AdvancedObsPadder,
    "general_stacker": GeneralStacker
}

_arg_transformers = { 
    "general_stacker": lambda **kwargs: {
        "obs": build_obs_builder_from_config(kwargs), 
        "stack_size": kwargs.get("stack_size", 15)
    }
}


register_obs_builder, build_obs_builder_from_config = build_component_factory(
    component_name="obs builder",
    builders=_builders,
    arg_transformers=_arg_transformers
)