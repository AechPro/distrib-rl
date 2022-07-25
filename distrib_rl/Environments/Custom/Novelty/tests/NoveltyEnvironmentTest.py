from distrib_rl.Environments.Custom.Novelty import Environment


def run_test():
    env = Environment()
    for i in range(9):
        env.reset()
        done = False
        while not done:
            action = i
            obs, rew, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        env.save("data/playbacks/test/1_{}.txt".format(i))
