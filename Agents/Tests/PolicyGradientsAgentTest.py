from Agents import PolicyGradientsAgent
import gym

class FakeDiscretePolicy(object):
    def get_action(self, obs):
        return 1, 1.0
    def get_output(self, obs):
        return [1]

def run_test():
    cfg = {}
    num_timesteps = 10
    env = gym.make("CartPole-v1")
    policy = FakeDiscretePolicy()
    agent = PolicyGradientsAgent(cfg)

    trajectories = agent.gather_timesteps(policy, env, num_timesteps)
    for traj in trajectories:
        print("Trajectory:\n{}".format(traj.serialize()))
    print()

    reward = agent.evaluate_policy(policy, env, num_timesteps=num_timesteps)
    print("Eval reward: {}".format(reward))

if __name__ == "__main__":
    run_test()