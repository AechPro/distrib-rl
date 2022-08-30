import importlib
import inspect

from distrib_rl.policies import policy_factory
from distrib_rl.gradient_optimization import GradientBuilder, gradient_optimizer_factory
from distrib_rl.agents import agent_factory
from distrib_rl.experience import ExperienceReplay
from distrib_rl.strategy import StrategyOptimizer
from distrib_rl.utils import AdaptiveOmega
from distrib_rl.policy_optimization.learners import *
import gym
import numpy as np
import random
import torch


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def build_env(cfg, existing_env=None):

    _register_custom_envs(cfg)

    if existing_env is None:
        env = gym.make(cfg["env_id"], new_step_api=True, **cfg.get("env_kwargs", {}))
    else:
        env = existing_env

    seed = cfg.get("seed", None)
    options = cfg.get("env_kwargs", None)

    env.reset(seed=seed, options=options)
    env.action_space.seed(cfg["seed"])
    return env


def build_vars(cfg, existing_env=None, env_space_shapes=None):
    seed = cfg["seed"]
    cfg["rng"] = np.random.RandomState(seed)
    device = cfg["device"]
    if env_space_shapes is None:
        env = build_env(cfg, existing_env=existing_env)
    else:
        env = None

    print("Set RNG seeds to {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    experience = ExperienceReplay(cfg)
    agent = agent_factory.get_from_cfg(cfg)

    models = policy_factory.get_from_cfg(
        cfg, env=env, env_space_shapes=env_space_shapes
    )

    policy = models["policy"]
    print(f"Policy params: {policy.num_params}")
    print(f"Policy input shape: {policy.input_shape}")
    print(f"Policy output shape: {policy.output_shape}")

    value_net = models["value_estimator"]
    print(f"Value net params: {value_net.num_params}")
    print(f"Value net input shape: {value_net.input_shape}")
    print(f"Value net output shape: {value_net.output_shape}")
    policy.to(device)
    value_net.to(device)
    models.clear()

    strategy_optimizer = StrategyOptimizer(cfg, policy)
    omega = AdaptiveOmega(cfg)

    gradient_builder = GradientBuilder(cfg)
    gradient_optimizers = gradient_optimizer_factory.get_from_cfg(cfg, value_net)
    value_gradient_optimizer = gradient_optimizers["value_gradient_optimizer"]
    gradient_optimizers.clear()

    gradient_optimizers = gradient_optimizer_factory.get_from_cfg(cfg, policy)
    novelty_gradient_optimizer = gradient_optimizers["novelty_gradient_optimizer"]
    policy_gradient_optimizer = gradient_optimizers["policy_gradient_optimizer"]
    gradient_optimizers.clear()

    policy_gradient_optimizer.omega = omega
    novelty_gradient_optimizer.omega = omega

    learner = DistribPPO(
        cfg,
        policy,
        value_net,
        policy_gradient_optimizer,
        value_gradient_optimizer,
        gradient_builder,
        omega,
    )
    # learner = PPONS(strategy_optimizer, cfg, policy, value_net, policy_gradient_optimizer, value_gradient_optimizer, gradient_builder, omega)

    return (
        env,
        experience,
        gradient_builder,
        policy_gradient_optimizer,
        value_gradient_optimizer,
        agent,
        policy,
        strategy_optimizer,
        omega,
        value_net,
        novelty_gradient_optimizer,
        learner,
    )


def _register_custom_envs(cfg):
    custom_envs = cfg.get("custom_envs", [])
    for custom_env in custom_envs:
        importlib.import_module(custom_env)


def _load_env(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def _is_configurable(func):
    return "config" in inspect.signature(func).parameters
