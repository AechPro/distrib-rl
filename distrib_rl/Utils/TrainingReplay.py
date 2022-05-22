from distrib_rl.PolicyOptimization.DistribPolicyGradients import Configurator
from distrib_rl.Utils import ConfigLoader


def run():
    cfg = ConfigLoader.load_config(file_name="test_distrib_config.json")
    opt_vars = Configurator.build_vars(cfg)
    policy = opt_vars[6]
    agent = opt_vars[5]
    env = opt_vars[0]
    policy.load("data/wordle/epoch_39200/actor")

    n_eps = 25
    for i in range(n_eps):
        rew = agent.evaluate_policy(policy, env)
        print("EPISODE REWARD: {}\n".format(rew))
