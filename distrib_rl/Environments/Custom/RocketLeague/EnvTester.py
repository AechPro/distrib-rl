from distrib_rl.Environments.Custom.RocketLeague import RLGymFactory
import numpy as np


if __name__ == "__main__":

    cfg = {"rlgym":{
            "tick_skip": 8,
            "team_size": 2,
            "game_speed": 1,
            "spawn_opponents": True,
            "env_id":2
          }}

    env = RLGymFactory.build_rlgym_from_config(cfg)


    try:
        ball_heights = []

        while True:
            obs, info = env.reset(return_info=True)
            gamestate = info["state"]
            ball_heights.append(gamestate.ball.position[2])
            done = False
            ep_rew = np.zeros(cfg["rlgym"]["team_size"]*2)
            n_steps = 0

            while not done:
                actions = [env.action_space.sample() for _ in range(cfg["rlgym"]["team_size"]*2)]
                obs, rew, done, gamestate = env.step(np.asarray(actions))
                ep_rew += rew
                n_steps += 1

            print("{:7.4f} | {:7.4f} | {:7.4f} | {:7.4f}".format(
                np.mean(ball_heights), np.std(ball_heights), min(ball_heights), max(ball_heights)
            ))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        env.close()
