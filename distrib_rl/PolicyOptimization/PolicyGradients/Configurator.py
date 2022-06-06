from distrib_rl.Policies import PolicyFactory
from distrib_rl.GradientOptimization import GradientOptimizerFactory, GradientBuilder
from distrib_rl.Agents import AgentFactory
from distrib_rl.Experience import ExperienceReplay
from distrib_rl.Strategy import StrategyOptimizer
from distrib_rl.Utils import AdaptiveOmega
from distrib_rl.PolicyOptimization.Learners import PPO, REINFORCE
import gym
import torch.optim
from distrib_rl.Environments.Custom import MinAtarWrapper
import numpy as np


def build_vars(cfg):
    cfg["rng"] = np.random.RandomState(cfg["seed"])
    #env = MinAtarWrapper(cfg_json["env_id"])

    env_name = cfg["env_id"].lower()
    if "rocket" in env_name:
        from distrib_rl.Environments.Custom.RocketLeague import RLGymFactory
        env = RLGymFactory.build_rlgym_from_config(cfg)
    else:
        env = gym.make(cfg["env_id"])

    env.seed(cfg["seed"])
    env.action_space.seed(cfg["seed"])
    experience = ExperienceReplay(cfg)
    agent = AgentFactory.get_from_cfg(cfg)

    models = PolicyFactory.get_from_cfg(cfg, env)
    policy = models["policy"]
    value_net = models["value_estimator"]
    models.clear()

    strategy_optimizer = StrategyOptimizer(cfg, policy, env)
    omega = AdaptiveOmega(cfg)

    gradient_builder = GradientBuilder(cfg)

    gradient_optimizers = GradientOptimizerFactory.get_from_cfg(cfg, policy)
    policy_gradient_optimizer = gradient_optimizers["policy_gradient_optimizer"]
    novelty_gradient_optimizer = gradient_optimizers["novelty_gradient_optimizer"]
    gradient_optimizers.clear()

    gradient_optimizers = GradientOptimizerFactory.get_from_cfg(cfg, value_net)
    value_gradient_optimizer = gradient_optimizers["value_gradient_optimizer"]
    gradient_optimizers.clear()

    policy_gradient_optimizer.omega = omega
    novelty_gradient_optimizer.omega = omega

    learner = PPO(cfg, policy, value_net, policy_gradient_optimizer, value_gradient_optimizer, gradient_builder, omega)


    return env, experience, gradient_builder, policy_gradient_optimizer, value_gradient_optimizer, agent, policy, \
           strategy_optimizer, omega, value_net, novelty_gradient_optimizer, learner
