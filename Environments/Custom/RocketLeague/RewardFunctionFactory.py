from rlgym.utils.reward_functions.combined_reward import CombinedReward
from rlgym.utils.reward_functions.common_rewards import AlignBallGoal, \
        BallYCoordinateReward, ConstantReward, EventReward, FaceBallReward, \
        LiuDistanceBallToGoalReward, LiuDistancePlayerToBallReward, \
        RewardIfBehindBall, RewardIfClosestToBall, RewardIfTouchedLast, \
        SaveBoostReward, TouchBallReward, VelocityBallToGoalReward, \
        VelocityPlayerToBallReward

from rlgym_tools.extra_rewards.anneal_rewards import AnnealRewards
from rlgym_tools.extra_rewards.diff_reward import DiffReward
from rlgym_tools.extra_rewards.distribute_rewards import DistributeRewards
from rlgym_tools.extra_rewards.jump_touch_reward import JumpTouchReward
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym_tools.extra_rewards.multi_model_rewards import MultiModelReward
from rlgym_tools.extra_rewards.multiply_rewards import MultiplyRewards
from rlgym_tools.extra_rewards.sequential_rewards import SequentialRewards

from Environments.Custom.RocketLeague.ComponentBuilderFactory import build_component_builder_factory

_builders = {
    "combined": CombinedReward,
    "align_ball_goal": AlignBallGoal,
    "ball_y_coordinate": BallYCoordinateReward,
    "constant": ConstantReward,
    "event": EventReward,
    "face_ball": FaceBallReward,
    "liu_distance_ball_to_goal": LiuDistanceBallToGoalReward,
    "liu_distance_player_to_ball": LiuDistancePlayerToBallReward,
    "save_boost": SaveBoostReward,
    "touch_ball": TouchBallReward,
    "if_behind_ball": RewardIfBehindBall,
    "if_closest_to_ball": RewardIfClosestToBall,
    "if_touched_last": RewardIfTouchedLast,
    "velocity_ball_to_goal": VelocityBallToGoalReward,
    "velocity_player_to_ball": VelocityPlayerToBallReward,
    "anneal_rewards": lambda **kwargs: AnnealRewards(*sum(zip([build_reward_fn_from_config(r) for r in kwargs["reward_functions"]], kwargs["weights"]), ())),
    "diff": DiffReward,
    "distribute": DistributeRewards,
    "jump_touch": JumpTouchReward,
    "kickoff": KickoffReward,
    "multi_model": MultiModelReward,
    "multiply": MultiplyRewards,
    "sequential": SequentialRewards
}

_arg_transformers = {
    "combined": lambda **kwargs: {
        "rewards": build_reward_fn_from_config(kwargs["reward_functions"]),
        "weights": tuple(kwargs["reward_weights"])
    },
    "diff": lambda **kwargs: {
        "reward": build_reward_fn_from_config(kwargs["reward_function"]),
        "negative_slope": kwargs.get("negative_slope", 0.1)
    },
    "distribute": lambda **kwargs: {
        "reward": build_reward_fn_from_config(kwargs["reward_function"]),
        "team_spirit": kwargs.get("team_spirit", 0.3)
    },
    "multi_model": lambda **kwargs: {
        "rewards": build_reward_fn_from_config(kwargs["reward_funcs"]),
        "model_map": kwargs["model_map"],
    },
    "multiply": lambda **kwargs: {
        "rewards": [build_reward_fn_from_config(f) for f in kwargs["reward_functions"]]
    },
    "sequential": lambda **kwargs: {
        "rewards": [build_reward_fn_from_config(f) for f in kwargs["rewards"]],
        "steps": kwargs["steps"]
    },
}


register_reward_function, build_reward_fn_from_config = build_component_builder_factory(
    "reward function",
    builders=_builders,
    arg_transformers=_arg_transformers,
    optional=False
)
