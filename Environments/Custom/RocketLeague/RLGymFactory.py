from Environments.Custom.RocketLeague.CustomEnvs import Default, SoloDefender
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState


def build_rlgym_from_config(config):
    cfg = config["rlgym"]

    action_parser = DiscreteAction()
    state_setter = DefaultState()
    obs_builder = DefaultObs()
    reward_fn = DefaultReward()
    terminal_conditions = [TimeoutCondition(225), GoalScoredCondition()]

    env_id = int(cfg["env_id"])

    if env_id == 0:
        action_parser = Default.DefaultActionParser()
        obs_builder = Default.DefaultObsBuilder()
        state_setter = Default.DefaultStateSetter()
        reward_fn = Default.DefaultRewardFunction()
        terminal_conditions = [Default.DefaultTerminalConditions()]

    elif env_id == 1:
        action_parser = SoloDefender.SoloDefenderActionParser()
        obs_builder = SoloDefender.SoloDefenderObsBuilder()
        state_setter = SoloDefender.SoloDefenderStateSetter()
        reward_fn = SoloDefender.SoloDefenderRewardFunction()
        terminal_conditions = [SoloDefender.SoloDefenderTerminalConditions()]

    return rlgym.make(
        game_speed=cfg["game_speed"],
        tick_skip=cfg["tick_skip"],
        spawn_opponents=cfg["spawn_opponents"],
        self_play=cfg["self_play"],
        team_size=cfg["team_size"],
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        state_setter=state_setter,
        action_parser=action_parser,
        use_injector=True,
        force_paging=True
    )