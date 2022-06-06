from distrib_rl.Policies import PolicyFactory
from distrib_rl.GradientOptimization import GradientOptimizerFactory, GradientBuilder
from distrib_rl.Agents import AgentFactory
from distrib_rl.Experience import ExperienceReplay
from distrib_rl.Strategy import StrategyOptimizer
from distrib_rl.Utils import AdaptiveOmega
from distrib_rl.PolicyOptimization.Learners import *
import gym
import numpy as np
import random
import torch


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def build_env(cfg, existing_env=None):
    env_name = cfg["env_id"].lower()
    if existing_env is None:
        if "rocket" in env_name:
            from distrib_rl.Environments.Custom.RocketLeague import RLGymFactory
            env = RLGymFactory.build_rlgym_from_config(cfg)
        else:
            env = gym.make(cfg["env_id"])
    elif "rocket" in env_name:
        from distrib_rl.Environments.Custom.RocketLeague import RLGymFactory
        env = RLGymFactory.build_rlgym_from_config(cfg, existing_env=existing_env)
    else:
        env = existing_env

    env.seed(cfg["seed"])
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
    agent = AgentFactory.get_from_cfg(cfg)

    models = PolicyFactory.get_from_cfg(cfg, env=env, env_space_shapes=env_space_shapes)

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
    gradient_optimizers = GradientOptimizerFactory.get_from_cfg(cfg, value_net)
    value_gradient_optimizer = gradient_optimizers["value_gradient_optimizer"]
    gradient_optimizers.clear()

    gradient_optimizers = GradientOptimizerFactory.get_from_cfg(cfg, policy)
    novelty_gradient_optimizer = gradient_optimizers["novelty_gradient_optimizer"]
    policy_gradient_optimizer = gradient_optimizers["policy_gradient_optimizer"]
    gradient_optimizers.clear()

    policy_gradient_optimizer.omega = omega
    novelty_gradient_optimizer.omega = omega

    learner = DistribPPO(cfg, policy, value_net, policy_gradient_optimizer, value_gradient_optimizer, gradient_builder, omega)
    # learner = PPONS(strategy_optimizer, cfg, policy, value_net, policy_gradient_optimizer, value_gradient_optimizer, gradient_builder, omega)

    return env, experience, gradient_builder, policy_gradient_optimizer, value_gradient_optimizer, agent, policy, \
           strategy_optimizer, omega, value_net, novelty_gradient_optimizer, learner
