import os
import torch
import numpy as np
from distrib_rl.GradientOptimization.Optimizers import GradientOptimizer


class TorchWrapper(GradientOptimizer):
    def __init__(self, policy, torch_optimizer, device):
        super().__init__(policy)
        self.device = device
        self.torch_optimizer = torch_optimizer

    def compute_update_step(self, gradient):
        device = self.device
        model = self.policy
        optim = self.torch_optimizer
        before = model.get_trainable_flat().copy()

        step = 0
        n_params = len(gradient)
        for p in model.parameters():
            param_size = np.prod(p.shape)
            next_step = step + param_size

            if next_step > n_params:
                grad_slice = [gradient[-1]]
            else:
                grad_slice = gradient[step:next_step]

            grad_tensor = torch.as_tensor(grad_slice, dtype=torch.float32).view_as(p).to(device)

            if p.grad is not None:
                p.grad.zero_()
                p.grad.add_(grad_tensor)

            step += param_size

        optim.step()
        after = model.get_trainable_flat(force_update=True).copy()
        update = after - before
        return update

    def cleanup(self):
        pass

    def save(self, file_path, name):
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        torch.save(self.torch_optimizer.state_dict(), os.path.join(file_path, name))

    def load(self, file_path, name):
        self.torch_optimizer.load_state_dict(torch.load(os.path.join(file_path, name)))

