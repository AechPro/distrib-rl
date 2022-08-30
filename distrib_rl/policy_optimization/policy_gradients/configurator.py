import importlib
import inspect
from distrib_rl.policies import policy_factory
from distrib_rl.gradient_optimization import GradientBuilder, gradient_optimizer_factory
from distrib_rl.agents import agent_factory
from distrib_rl.experience import ExperienceReplay
from distrib_rl.strategy import StrategyOptimizer
from distrib_rl.utils import AdaptiveOmega
from distrib_rl.policy_optimization.learners import PPO
import gym
import numpy as np


def build_vars(cfg):
    cfg["rng"] = np.random.RandomState(cfg["seed"])

    _register_custom_envs(cfg)

    env = gym.make(cfg["env_id"])

    seed = cfg.get("seed", None)
    options = cfg.get("env_kwargs", None)

    env.reset(seed=seed, options=options)

    env.seed(cfg["seed"])
    env.action_space.seed(cfg["seed"])
    experience = ExperienceReplay(cfg)
    agent = agent_factory.get_from_cfg(cfg)

    models = policy_factory.get_from_cfg(cfg, env)
    policy = models["policy"]
    value_net = models["value_estimator"]
    models.clear()

    strategy_optimizer = StrategyOptimizer(cfg, policy, env)
    omega = AdaptiveOmega(cfg)

    gradient_builder = GradientBuilder(cfg)

    gradient_optimizers = gradient_optimizer_factory.get_from_cfg(cfg, policy)
    policy_gradient_optimizer = gradient_optimizers["policy_gradient_optimizer"]
    novelty_gradient_optimizer = gradient_optimizers["novelty_gradient_optimizer"]
    gradient_optimizers.clear()

    gradient_optimizers = gradient_optimizer_factory.get_from_cfg(cfg, value_net)
    value_gradient_optimizer = gradient_optimizers["value_gradient_optimizer"]
    gradient_optimizers.clear()

    policy_gradient_optimizer.omega = omega
    novelty_gradient_optimizer.omega = omega

    learner = PPO(
        cfg,
        policy,
        value_net,
        policy_gradient_optimizer,
        value_gradient_optimizer,
        gradient_builder,
        omega,
    )

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
    if ":" in name:
        mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def _is_configurable(func):
    return "config" in inspect.signature(func).parameters
