import numpy as np
from rlgym.utils import common_values, math
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, EventReward


class KickoffRewardFunction(RewardFunction):
    def __init__(self):
        self.rf = VelocityPlayerToBallReward()
        self.prev_boosts = {}

    def reset(self, initial_state: GameState):
        self.rf.reset(initial_state)
        self.prev_boosts = {p.car_id:p.boost_amount for p in initial_state.players}

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.team_num == common_values.ORANGE_TEAM:
            ball = state.inverted_ball
        else:
            ball = state.ball

        c1 = self.rf.get_reward(player, state, previous_action)
        c2 = 0
        c3 = 0
        c4 = 0
        if not player.has_flip:
            c2 = 0.25 / 100

        if any(p.ball_touched for p in state.players):
            c3 = ball.linear_velocity[1]

        if player.boost_amount > self.prev_boosts[player.car_id]:
            c4 = 0.25

        rew = c1 / 30 + c2 + c3 * 10 / common_values.BALL_MAX_SPEED + c4

        self.prev_boosts[player.car_id] = player.boost_amount
        return rew