import numpy as np
from distrib_rl.GradientOptimization.Optimizers import GradientOptimizer
from distrib_rl.Utils import MathHelpers as math


class DynamicSGD(GradientOptimizer):
    def __init__(
        self,
        policy,
        max_update_scale=1.0,
        min_update_scale=0.2,
        step_size=3e-4,
        omega_min=0,
        omega_max=1,
    ):
        super().__init__(policy)

        self.max_update_scale = max_update_scale
        self.min_update_scale = min_update_scale
        self.step_size = step_size
        self.update_magnitude = np.sqrt(policy.num_params)

        self.omega_min = omega_min
        self.omega_max = omega_max
        self.omega = None

        self.prev_grads = []
        print("DSGD HAS HARD-CODED OMEGA TO 0 EVERY UPDATE")

    def compute_update_step(self, gradient):
        norm = 1  # np.linalg.norm(gradient)
        step_size = self.step_size
        update_magnitude = self.update_magnitude

        if norm == 0:
            norm = 1

        if self.omega is not None:
            w = 0  # self.omega.omega
            modifier = math.apply_affine_map(
                w,
                from_min=self.omega_min,
                from_max=self.omega_max,
                to_min=self.min_update_scale,
                to_max=self.max_update_scale,
            )
        else:
            modifier = norm / update_magnitude

        eta = -modifier * step_size * update_magnitude / norm
        return gradient * eta

    def save(self, file_path, name):
        pass

    def load(self, file_path, name):
        pass

    def cleanup(self):
        pass
