from distrib_rl.Environments.Custom.RocketLeague.CustomEnvs import Default, SoloDefender, Kickoff
import rlgym
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from distrib_rl.Environments.Custom.RocketLeague.ActionParserFactory import build_action_parser_from_config
from distrib_rl.Environments.Custom.RocketLeague.ObsBuilderFactory import build_obs_builder_from_config
from distrib_rl.Environments.Custom.RocketLeague.RewardFunctionFactory import build_reward_fn_from_config
from distrib_rl.Environments.Custom.RocketLeague.StateSetterFactory import build_state_setter_from_config
from distrib_rl.Environments.Custom.RocketLeague.TerminalConditionsFactory import build_terminal_conditions_from_config


def build_rlgym_from_config(config, existing_env=None):
    cfg = config["rlgym"]

    action_parser = DiscreteAction()
    state_setter = DefaultState()
    obs_builder = DefaultObs()
    reward_fn = DefaultReward()
    terminal_conditions = [TimeoutCondition(225), GoalScoredCondition()]

    if cfg.get("env_id", None):
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
        
        elif env_id == 2:
            action_parser = Kickoff.NectoActionParser()
            obs_builder = Kickoff.KickoffObsBuilder()
            state_setter = Kickoff.KickoffStateSetter()
            reward_fn = Kickoff.KickoffRewardFunction()
            terminal_conditions = [Kickoff.KickoffTerminalConditions()]

    if cfg.get("action_parser", False):
        action_parser = build_action_parser_from_config(cfg["action_parser"])
    if cfg.get("obs_builder", False):
        obs_builder = build_obs_builder_from_config(cfg["obs_builder"])
    if cfg.get("state_setters", False):
        state_setter = build_state_setter_from_config(cfg["state_setters"])
    if cfg.get("rewards", False):
        reward_fn = build_reward_fn_from_config(cfg["rewards"])
    if cfg.get("terminal_conditions", False):
        terminal_conditions = build_terminal_conditions_from_config(cfg["terminal_conditions"])

    if existing_env:
        match = existing_env._match
        match.__init__(
            game_speed=cfg["game_speed"],
            tick_skip=cfg["tick_skip"],
            spawn_opponents=cfg["spawn_opponents"],
            team_size=cfg["team_size"],
            terminal_conditions=terminal_conditions,
            reward_function=reward_fn,
            obs_builder=obs_builder,
            state_setter=state_setter,
            action_parser=action_parser
        )

        existing_env.observation_space = match.observation_space
        existing_env.action_space = match.action_space

        # del clears references so we don't accidentally mangle rlgym state
        # (probably)
        o = existing_env.reset()
        del o
        return existing_env

    return rlgym.make(
        game_speed=cfg["game_speed"],
        tick_skip=cfg["tick_skip"],
        spawn_opponents=cfg["spawn_opponents"],
        team_size=cfg["team_size"],
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        state_setter=state_setter,
        action_parser=action_parser,
        use_injector=True,
        force_paging=True,
        auto_minimize=cfg.get("auto_minimize", True)
    )
