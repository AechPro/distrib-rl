import numpy as np


class GradientOptimizer(object):
    def __init__(self, policy):
        self.steps = 0
        self.policy = policy

    def compute_update(self, theta, gradient):
        self.steps += 1
        update = self.compute_update_step(gradient)
        output = np.add(update, theta)

        del update
        return output

    def compute_update_step(self, gradient):
        """
        Function to compute the update step for a policy with some gradient.
        :param gradient: Gradient over which to optimize.
        :return: Computed update step.
        """
        raise NotImplementedError

    def cleanup(self):
        raise NotImplementedError

    def save(self, file_path, name):
        raise NotImplementedError

    def load(self, file_path, name):
        raise NotImplementedError