"""
    File name: Adam.py
    Author: Matthew Allen

    Description:
        An implementation of the Adam optimizer. This is pretty much just a copy of
        the implementation by OpenAI.

        See: https://arxiv.org/abs/1412.6980
"""
import numpy as np
import os
from distrib_rl.GradientOptimization.Optimizers import GradientOptimizer


class Adam(GradientOptimizer):
    def __init__(self, policy, step_size=3e-4, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(policy)

        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        dim = policy.num_params
        self.m = np.zeros(dim, dtype=np.float32)
        self.v = np.zeros(dim, dtype=np.float32)

    def compute_update_step(self, gradient):
        a = (
            self.step_size
            * np.sqrt(1 - self.beta2**self.steps)
            / (1 - self.beta1**self.steps)
        )
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient * gradient)
        update_step = -a * self.m / (np.sqrt(self.v) + self.epsilon)

        return update_step

    def cleanup(self):
        del self.m
        del self.v

    def save(self, file_path, name):
        with open(os.path.join(file_path, name), "w") as f:
            for arg in np.ravel(self.m):
                f.write("{} ".format(arg))
            f.write("\n")
            for arg in np.ravel(self.v):
                f.write("{} ".format(arg))
            f.write("\n")
            f.write("{}".format(self.steps))

    def load(self, file_path, name):
        with open(os.path.join(file_path, name), "r") as f:
            lines = f.readlines()
            m = []
            v = []

            for arg in lines[0].split(" "):
                try:
                    m.append(float(arg.strip()))
                except:
                    continue

            for arg in lines[1].split(" "):
                try:
                    v.append(float(arg.strip()))
                except:
                    continue

            self.steps = float(lines[2].strip())
            self.v = np.asarray(v).reshape(self.v.shape).astype("float32")
            self.m = np.asarray(m).reshape(self.m.shape).astype("float32")
