import torch
import numpy as np
from distrib_rl.Agents import AgentFactory
from distrib_rl.Strategy import StrategyPoint


class StrategyOptimizer(object):
    def __init__(self, cfg, policy):
        self.cfg = cfg
        self.device = cfg["device"]
        self.max_history_size = cfg["strategy"]["max_history_size"]
        self.num_frames = cfg["strategy"]["num_frames"]
        self.steps_per_eval = cfg["strategy"]["steps_per_eval"]
        self.num_fd_perturbations = cfg["strategy"]["num_fd_perturbations"]
        self.fd_noise_std = cfg["strategy"]["fd_noise_std"]

        self.policy = policy
        self.agent = AgentFactory.get_from_cfg(cfg)

        self.strategy_history = []
        self.beh_tensor = torch.as_tensor([])
        self.frames = []
        self.current_step = 0

        self.gradient = np.zeros(policy.num_params).astype(np.float32)

    def update(self):
        if self.frames is None or len(self.frames) == 0:
            return

        if self.current_step % self.steps_per_eval == 0:
            self.eval_history()

        self.add_current_policy()
        self.current_step += 1

    def set_from_server(self, strategy_frames, strategy_history):
        self.beh_tensor = torch.as_tensor(strategy_history, dtype=torch.float32).to(
            self.device
        )
        self.frames = torch.as_tensor(strategy_frames, dtype=torch.float32).to(
            self.device
        )

    @torch.no_grad()
    def add_current_policy(self):
        point = StrategyPoint(self.policy.get_trainable_flat(force_update=True).copy())
        self.strategy_history.append(point)

        while len(self.strategy_history) > self.max_history_size:
            point = self.strategy_history.pop(
                np.random.randint(0, len(self.strategy_history) - 1)
            )
            point.cleanup()
            del point

    @torch.no_grad()
    def eval_history(self):
        policy = self.policy
        frames = self.frames

        behs = []
        for point in self.strategy_history:
            point.compute_strategy(policy, frames)
            behs.append(point.strategy)

        self.beh_tensor = torch.as_tensor(behs, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def get_novelty_gradient(self):
        num = self.num_fd_perturbations
        epsilon_std = self.fd_noise_std
        policy = self.policy
        rng = self.cfg["rng"]
        beh_hist = self.beh_tensor
        frames = self.frames
        self.gradient = np.zeros(policy.num_params).astype(np.float32)
        gradient = self.gradient

        orig = policy.get_trainable_flat(force_update=True).copy()
        length = len(orig)

        if len(beh_hist) < 2:
            return np.zeros(length)

        novelties = []
        epsilons = []

        for i in range(num):
            noise = rng.randn(length).astype(np.float32)
            flat = np.add(orig, np.multiply(noise, epsilon_std))
            policy.set_trainable_flat(flat)

            beh = policy.get_output(frames).flatten()
            beh_dists = (beh_hist - beh.view(1, beh.shape[0])).norm(dim=1)

            novelty = beh_dists.cpu().min().item()
            novelties.append(novelty)

            vecnorm = np.linalg.norm(noise)
            epsilons.append(noise / (vecnorm * vecnorm))

        policy.set_trainable_flat(orig)
        novs = np.asarray(novelties, dtype=np.float32)
        print(RLMath.compute_array_stats(novs))

        mean = novs.mean()
        std = novs.std()
        if std == 0:
            std = 1
        cost = (novs - mean) / std

        np.dot(cost / num, epsilons, out=gradient)
        return gradient

    def get_novelty_gradient_bprop(self):
        if len(self.beh_tensor) < 2:
            return False

        beh = self.policy.get_output(self.frames).flatten()
        dists = (self.beh_tensor - beh).norm(dim=1)
        dist = -dists.mean()
        dist.backward()
        return True

    @torch.no_grad()
    def get_nearest_policy(self, frames):
        policy = self.policy

        behs = []
        for point in self.strategy_history:
            point.compute_strategy(policy, frames)
            behs.append(point.strategy)

        beh_hist = torch.as_tensor(behs, dtype=torch.float32).to(self.device)

        beh = policy.get_output(frames).flatten()
        beh_dists = (beh_hist - beh.view(1, beh.shape[0])).norm(dim=1)
        argmin = beh_dists.cpu().argmin().item()

        return self.strategy_history[argmin].flat
        # return behs[argmin]

    @torch.no_grad()
    def compute_policy_novelty(self):
        if len(self.beh_tensor) < 2:
            return 0
        beh = self.policy.get_output(self.frames)
        beh_dists = (self.beh_tensor - beh.flatten()).norm(dim=1)
        return beh_dists.min().item()

    def set_frames(self, frames):
        self.frames = torch.as_tensor(frames, dtype=torch.float32).to(self.device)

    def serialize(self):
        if self.frames is None or len(self.frames) == 0:
            return np.asarray([0]), np.asarray([0])

        return self.frames.cpu().numpy(), self.beh_tensor.cpu().numpy()
