import numpy as np
import os


class AdaptiveOmega(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.omega = cfg["adaptive_omega"]["default"]
        self.improvement_threshold = cfg["adaptive_omega"]["mean_threshold"]
        self.reward_history_size = cfg["adaptive_omega"]["reward_history_size"]
        self.min_omega = cfg["adaptive_omega"]["min_value"]
        self.max_omega = cfg["adaptive_omega"]["max_value"]

        self.reward_history = []
        self.up_steps = 0
        self.down_steps = 0

        self.steps_to_max = 200
        self.steps_to_min = 15

        self.increase = 1 / self.steps_to_max
        self.decrease = 1 / self.steps_to_min

        self.decrease_start = 0
        self.increase_start = 0

        self.benchmark = -np.inf

        self.flag = False
        self.max = -np.inf

    def step(self, theta_reward):
        if theta_reward is None:
            return

        self.advance_reward_history(theta_reward)
        self.adapt_omega(theta_reward)

    def adapt_omega(self, theta_reward):
        if len(self.reward_history) == 0:
            return
        mean_reward = float(np.mean(self.reward_history))

        mean_reward = round(mean_reward, 5)
        theta_reward = round(theta_reward, 5)

        if mean_reward < 0:
            mean_reward /= self.improvement_threshold
        else:
            mean_reward *= self.improvement_threshold

        if theta_reward > mean_reward:
            self.omega = max(self.omega - self.decrease, self.min_omega)
        else:
            self.omega = min(self.omega + self.increase, self.max_omega)

    def save(self, path):
        with open(os.path.join(path, "omega.dat"), "w") as f:
            f.write("{}".format(self.omega))

    def advance_reward_history(self, reward):
        self.reward_history.append(reward)
        if len(self.reward_history) > self.reward_history_size:
            oldest = self.reward_history.pop(0)
            del oldest

    def cleanup(self):
        del self.reward_history
        del self.cfg

        self.cfg = None
        self.omega = 0
        self.improvement_threshold = 1
        self.reward_history_size = 0
        self.min_omega = 0
        self.max_omega = 0
        self.reward_history = []
