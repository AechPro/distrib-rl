import os
from typing import Optional, Tuple, Union
from numpy import ndarray

from rlgym.envs import Match
from rlgym.gym import Gym as BaseRLGymEnvironment
from rlgym.gamelaunch import LaunchPreference

from gym import Env
from gym.utils import seeding

from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.state_setters import DefaultState
from distrib_rl.Environments.Custom.RocketLeague.ActionParserFactory import build_action_parser_from_config
from distrib_rl.Environments.Custom.RocketLeague.ObsBuilderFactory import build_obs_builder_from_config
from distrib_rl.Environments.Custom.RocketLeague.RewardFunctionFactory import build_reward_function_from_config
from distrib_rl.Environments.Custom.RocketLeague.StateSetterFactory import build_state_setter_from_config
from distrib_rl.Environments.Custom.RocketLeague.TerminalConditionsFactory import build_terminal_conditions_from_config

_match_config_parsers = {
    "action_parser": build_action_parser_from_config,
    "obs_builder": build_obs_builder_from_config,
    "state_setter": build_state_setter_from_config,
    "reward_function": build_reward_function_from_config,
    "terminal_conditions": build_terminal_conditions_from_config,
}

_match_kwarg_names = [
    "action_parser",
    "obs_builder",
    "reward_function",
    "state_setter",
    "terminal_conditions",
    "team_size",
    "tick_skip",
    "game_speed",
    "gravity",
    "boost_consumption",
    "spawn_opponents"
]

_env_kwarg_names = [
    "launch_preference",
    "use_injector",
    "force_paging",
    "raise_on_crash",
    "auto_minimize"
]

class RLGymEnvironment(BaseRLGymEnvironment):
    """The main Rocket League Gym class. It encapsulates the process of managing
    the RLGym environment according to a dynamic, declarative configuration.

    The methods are accessed publicly as "step", "reset", etc...
    """

    _match: Match

    def __init__(self, **kwargs):
        self._config = kwargs
        match_kwargs = self._parse_match_kwargs(kwargs)
        self._match = Match(**match_kwargs)
        env_kwargs = self._parse_env_kwargs(kwargs)

        super().__init__(self._match, **env_kwargs)

        self.observation_space = self._match.observation_space
        self.action_space = self._match.action_space


    def step(
        self, action: ndarray
    ) -> Union[
        Tuple[ndarray, float, bool, bool, dict], Tuple[ndarray, float, bool, dict]
    ]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns either a tuple `(observation, reward, terminated, truncated, info)`, or a tuple
        (observation, reward, done, info). The latter is deprecated and will be removed in future versions.

        Args:
            action (ActType): an action provided by the agent

        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.

            (deprecated)
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
        """

        obs, reward, done, info = super().step(action)

        # Note: RLGym doesn't return a value for terminated, so we'll just
        # assume it's False
        return obs, reward, done, False, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[ndarray, Tuple[ndarray, dict]]:
        """Resets the environment to an initial state and returns the initial observation.

        This method can reset the environment's random number generator(s) if ``seed`` is an integer or
        if the environment has not yet initialized a random number generator.
        If the environment already has a random number generator and :meth:`reset` is called with ``seed=None``,
        the RNG should not be reset. Moreover, :meth:`reset` should (in the typical use case) be called with an
        integer seed right after initialization and then never again.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG.
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            return_info (bool): If true, return additional information along with initial observation.
                This info should be analogous to the info returned in :meth:`step`
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)


        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (optional dictionary): This will *only* be returned if ``return_info=True`` is passed.
                It contains auxiliary information complementing ``observation``. This dictionary should be analogous to
                the ``info`` returned by :meth:`step`.
        """

        if options is not None:
            self._config = options

            if self._match is not None:
                match_kwargs = self._parse_match_kwargs(self._config)
                self._match.__init__(**match_kwargs)

                self.observation_space = self._match.observation_space
                self.action_space = self._match.action_space

        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        return super().reset(return_info=return_info)

    def _parse_match_kwargs(self, config):
        """Parses the config and returns the kwargs for the Match constructor

        Args:
            config (dict): The config to parse.

        Returns:
            Dict: The kwargs for the Match constructor.
        """
        kwargs = {
            "reward_function": DefaultReward(),
            "terminal_conditions": [TimeoutCondition(225), GoalScoredCondition()],
            "obs_builder": DefaultObs(),
            "action_parser": DiscreteAction(),
            "state_setter": DefaultState(),
            "team_size": 1,
            "tick_skip": 8,
            "game_speed": 100,
            "gravity": 1,
            "boost_consumption": 1,
            "spawn_opponents": False
        }

        for key, value in config.items():
            if key in _match_config_parsers:
                kwargs[key] = _match_config_parsers[key](value)
            elif key in _match_kwarg_names:
                kwargs[key] = value
            elif key in _env_kwarg_names:
                pass
            else:
                raise ValueError(f"Unknown config key for environment `RocketLeague-v0`: {key}")

        return kwargs

    def _parse_env_kwargs(self, config):
        # note: the values for `use_injector`, `force_paging`, and
        # `auto_minimize` are the opposite of the actual RLGym defaults, however
        # they are consistent with the old implementation in distr-rl for the
        # RLGym environment
        kwargs = {
            "pipe_id": os.getpid(),
            "launch_preference": LaunchPreference.EPIC,
            "use_injector": True,
            "force_paging": True,
            "raise_on_crash": False,
            "auto_minimize": True
        }

        for key, value in config.items():
            if key in _env_kwarg_names:
                kwargs[key] = value
            elif key in _match_kwarg_names:
                pass
            else:
                raise ValueError(f"Unknown config key for environment `RocketLeague-v0`: {key}")

        return kwargs

