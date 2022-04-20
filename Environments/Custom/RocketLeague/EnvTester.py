

if __name__ == "__main__":
    from Environments.Custom.RocketLeague import RLGymFactory

    cfg = {"rlgym":
          {
            "tick_skip": 8,
            "team_size": 1,
            "game_speed": 2,
            "self_play": False,
            "spawn_opponents": False,
            "env_id":1
          }}

    env = RLGymFactory.build_rlgym_from_config(cfg)

    try:
        while True:
            env.reset()
            done = False
            ep_rew = 0
            n_steps = 0

            while not done:
                act = env.action_space.sample()
                obs, rew, done, _ = env.step(act)
                ep_rew += rew
                n_steps += 1

            print(ep_rew, "|", n_steps)
    except:
        env.close()