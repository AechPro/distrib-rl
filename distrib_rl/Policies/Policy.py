import numpy as np
import torch
import torch.nn as nn
from functools import partial
from distrib_rl.Utils.Torch import TorchFunctions
import os


class Policy(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        self.num_params = 0
        self.model = None
        self.flat = None
        self.device = device
        self.input_shape = None
        self.output_shape = None

    def get_backprop_data(self, obs, acts):
        raise NotImplementedError

    def get_action(self, obs, deterministic=False):
        raise NotImplementedError

    def build_model(self, model_json, input_shape, output_shape):
        raise NotImplementedError

    def _init_params(self):
        """
        Orthogonal initialization procedure. This is super random, blame OpenAI.
        :return:
        """
        if self.output_shape == 1:
            gain = 1
        else:
            gain = self.cfg["init_std"]

        n_trainable_layers = 0
        for layer in self.model:
            trainable = False
            for p in layer.parameters():
                if p.requires_grad:
                    trainable = True
                    break
            if trainable:
                n_trainable_layers += 1

        i = 0
        for layer in self.model:
            trainable = False
            for p in layer.parameters():
                if p.requires_grad:
                    trainable = True
                    break
            if trainable:
                if i < n_trainable_layers - 1:
                    layer.apply(
                        partial(TorchFunctions.init_weights_orthogonal, gain=gain)
                    )
                elif gain != 1:
                    layer.apply(
                        partial(
                            TorchFunctions.init_weights_orthogonal,
                            gain=self.cfg["action_init_std"],
                        )
                    )
                i += 1

    def get_output(self, obs):
        if type(obs) is not torch.Tensor:
            obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        if obs.device != self.device:
            obs = obs.to(self.device)

        return self.model(obs)

    def get_trainable_flat(self, force_update=False):
        if self.flat is None or force_update:
            self.flat = (
                torch.nn.utils.parameters_to_vector(self.parameters())
                .cpu()
                .detach()
                .numpy()
                .astype(np.float32)
            )
            self.num_params = len(self.flat)
        return self.flat

    def set_trainable_flat(self, flat):
        tensor = torch.as_tensor(flat, dtype=torch.float32).to(self.device)
        torch.nn.utils.vector_to_parameters(tensor, self.parameters())
        self.get_trainable_flat(force_update=True)

    def set_gradient_from_flat(self, flat):
        idx = 0
        for p in self.parameters():
            step = np.prod(p.shape)
            grad = flat[idx : idx + step]
            grad = torch.as_tensor(grad, dtype=torch.float32).view_as(p)
            p.backward(-grad)
            idx += step

    def save(self, file_path, name):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        flat = self.get_trainable_flat(force_update=True)
        print("POLICY SAVE FUNCTION", os.path.join(file_path, name))
        np.save(os.path.join(file_path, name), flat)

    def load(self, file_path, name):
        flat = np.load(os.path.join(file_path, name)).astype(np.float32)
        self.set_trainable_flat(flat)

    def zero_grad(self, set_to_none=False):
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def to(self, device):
        self.device = device
        super().to(device)
