from distrib_rl.Environments.Custom.Novelty import Environment


def run_test():
    env = Environment()
    for i in range(9):
        env.reset()
        done = False
        while not done:
            action = i
            obs, rew, done, _ = env.step(action)

        env.save("data/playbacks/test/1_{}.txt".format(i))