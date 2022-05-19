import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder

LARGE_BOOST_MASK = np.array([
    0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
    0.0
])

class DefaultWithTimeoutsObsBuilder(ObsBuilder):
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi, tick_skip=8):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        :param tick_skip: how many ticks are skipped between updates - used for
                          determining how much game time has transpired
        """
        super().__init__()
        self._state = None

        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.n_players = None

        self.TIMER_DECAY = tick_skip / 1200.0

    def reset(self, initial_state: GameState):
        self.n_players = len(initial_state.players)
        self._state = initial_state
        self.boost_pad_timers: np.ndarray = np.zeros(GameState.BOOST_PADS_LENGTH, dtype=np.float32)
        self.inverted_boost_pad_timers = np.zeros(GameState.BOOST_PADS_LENGTH, dtype=np.float32)
        self.waiting_mask = np.zeros(GameState.BOOST_PADS_LENGTH, dtype=np.bool8)
        self.demo_timers = np.zeros(self.n_players, dtype=np.float32)

    def _update_boost_timers(self, curr_boost_state, prev_boost_state):
        pad_updates =  curr_boost_state - prev_boost_state
        grabbed = (pad_updates == -1)
        spawned = (pad_updates == 1)

        self.waiting_mask = (self.waiting_mask | grabbed) & ~spawned

        self.boost_pad_timers[~self.waiting_mask] = 0
        self.boost_pad_timers[self.waiting_mask] -= self.TIMER_DECAY

        rolled_over = self.boost_pad_timers < 0

        self.boost_pad_timers[rolled_over | spawned] = 0.4 + 0.6 * LARGE_BOOST_MASK[rolled_over | spawned]
        self.inverted_boost_pad_timers[:] = self.boost_pad_timers[::-1]

    def _update_demo_timers(self, players):
        demos = np.array([p.is_demoed for p in players], dtype=np.float32)
        new_demos = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers -= self.TIMER_DECAY
        self.demo_timers *= demos

    def _step_state(self, state: GameState):
        if state == self._state:
            return

        if self._state == None:
            self.reset(state)

        self._update_boost_timers(state.boost_pads, self._state.boost_pads)
        self._update_demo_timers(state.players)
        self._state = state

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if (state != self._state):
            self._step_state(state)
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
            pad_timers = self.inverted_boost_pad_timers
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads
            pad_timers = self.boost_pad_timers

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action, pads, pad_timers]

        player_index = state.players.index(player)
        self._add_player_to_obs(obs, player, player_index, inverted)

        allies = []
        enemies = []

        for i, other in enumerate(state.players):
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, i, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, player_index: int, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [
                player.boost_amount,
                int(player.on_ground),
                int(player.has_flip),
                int(player.is_demoed),
                self.demo_timers[player_index]
            ]
        ])

        return player_car
