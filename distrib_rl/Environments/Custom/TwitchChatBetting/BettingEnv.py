import numpy as np
from gym.spaces import Box
from gym import Env

class BettingEnv(Env):
    def __init__(self):
        self.rng = np.random.RandomState(123)
        self.max_bets = 100
        self.chat_bias = 0.25

        self.current_probs = [1,0]
        self.biased_probs = [1,0]
        self.n_bets = 0
        self.default_money = 1
        self.money = self.default_money

        self.action_space = Box(0, 1, (2,))
        self.observation_space = Box(-np.inf, np.inf, (3,))

    def reset(self):
        self.n_bets = 0
        self.money = self.default_money
        self._create_probs()

        return self._build_obs()

    def step(self, action):
        action = np.add(action, 1)
        action = np.divide(action, sum(action))
        choice = np.argmax(action)
        bet = action[choice]*self.money

        profit = -bet
        roll = self.rng.choice((0, 1), p=self.current_probs)
        if roll == choice:
            profit += bet * (1 + min(self.biased_probs) / max(self.biased_probs))

        rew = (self.money + profit) / (self.money * self.max_bets)
        self.money += profit

        obs = self._build_obs()
        done = self.n_bets >= self.max_bets
        self.n_bets += 1

        if self.money < 0.1*self.default_money:
            rew = -10
            done = True
        return obs, rew, done, {}

    def render(self, mode="human"):
        pass

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def _create_probs(self):
        true_prob = self.rng.uniform(0, 1)
        self.current_probs = (true_prob, 1 - true_prob)

        bias = self.rng.uniform(0, self.chat_bias)
        biased_probs = (true_prob + bias, 1 - true_prob)
        self.biased_probs = np.divide(biased_probs, sum(biased_probs))

    def _build_obs(self):
        return [arg for arg in self.biased_probs] + [self.money / 1e3]