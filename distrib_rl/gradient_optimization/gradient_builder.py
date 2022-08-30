import numpy as np


class GradientBuilder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.gradient = None

    def contribute_gradient_from_flat(self, flat, coef, force_norm=True):
        if self.gradient is None:
            self.gradient = np.zeros(len(flat)).astype(np.float32)

        if type(flat) not in (np.ndarray, np.array):
            flat = np.asarray(flat).astype(np.float32)

        norm = np.linalg.norm(flat)
        if norm == 0:
            norm = 1

        if not force_norm and norm < coef:
            self.gradient += flat
        else:
            self.gradient += flat * coef / norm

    def contribute_gradient_from_model(self, model, coef, force_norm=True):
        flat = self._get_flat_gradient(model)
        self.contribute_gradient_from_flat(flat, coef, force_norm=force_norm)
        model.zero_grad()

    def update_model(self, model, gradient_optimizer):
        theta = model.get_trainable_flat(force_update=True).copy()
        next_theta = gradient_optimizer.compute_update(theta, self.gradient)
        model.set_trainable_flat(next_theta)
        model.zero_grad()
        self.reset()
        del next_theta

    def reset(self):
        del self.gradient
        self.gradient = None

    def _get_flat_gradient(self, model):
        flat = []
        for p in model.parameters():
            p.detach()
            if p.grad is not None:
                grad = p.grad.cpu().detach().data.numpy().ravel().astype(np.float32)
            else:
                grad = np.zeros(p.shape).ravel().astype(np.float32)

            flat = np.concatenate((flat, grad))
        return flat
