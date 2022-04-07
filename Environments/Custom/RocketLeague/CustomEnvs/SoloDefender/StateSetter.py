from rlgym.utils.state_setters import StateSetter, StateWrapper
from rlgym.utils import common_values, math
import numpy as np


class SoloDefenderStateSetter(StateSetter):
    GOAL_CORNERS = [(-893, 5120, 642)]# top right?
    def __init__(self):
        super().__init__()
        self.rng = np.random.RandomState(123)

    def reset(self, state_wrapper: StateWrapper):
        self._spawn_ball(state_wrapper)
        self._spawn_car(state_wrapper)
        return

    def _spawn_ball(self, state_wrapper: StateWrapper):
        start_point = (0, -5120/2, 642.775/2)
        end_point = common_values.BLUE_GOAL_CENTER

        x_offset = self.rng.uniform(-893 + common_values.BALL_RADIUS, 893 - common_values.BALL_RADIUS)
        y_offset = 800
        z_offset = self.rng.uniform(-602.775 / 2, 602.775 / 2)
        end_point = np.add(end_point, (x_offset, y_offset, z_offset))

        # point at center of goal
        velocity_vector = np.subtract(end_point, start_point)

        # set speed
        velocity_vector /= np.linalg.norm(velocity_vector)
        velocity_vector *= self.rng.uniform(common_values.BALL_MAX_SPEED/8, common_values.BALL_MAX_SPEED/2)

        state_wrapper.ball.set_pos(*start_point)
        state_wrapper.ball.set_lin_vel(*velocity_vector)

    def _spawn_car(self, state_wrapper: StateWrapper):
        x,y,_ = common_values.BLUE_GOAL_CENTER
        z = 0
        y += 120
        state_wrapper.cars[0].set_pos(x,y,z)
        state_wrapper.cars[0].set_rot(yaw=self.rng.uniform(0, np.pi*2))
        state_wrapper.cars[0].boost = 100