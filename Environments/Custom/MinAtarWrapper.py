from minatar import Environment
import gym
import gym.spaces
import numpy as np
import time


class MinAtarWrapper(gym.Env):
    def __init__(self, env_name):
        env = Environment(env_name)
        self.observation_space = gym.spaces.Box(0, 1, (np.prod(env.state_shape()),))
        self.action_space = gym.spaces.Discrete(env.num_actions())

        self.env = env

    def reset(self):
        self.env.reset()
        return self._wrap_state(self.env.state())

    def step(self, action):
        reward, terminated = self.env.act(action)
        next_state = self._wrap_state(self.env.state())
        return next_state, reward, terminated, {}

    def render(self):
        self.env.display_state(time=16)

    def _wrap_state(self, state):
        return state.ravel()

    def seed(self, seed):
        pass