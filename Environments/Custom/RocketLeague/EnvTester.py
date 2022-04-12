

if __name__ == "__main__":
    from Environments.Custom.RocketLeague import RLGymFactory

    env_id = 1
    cfg = {
          "rlgym":
          {
            "tick_skip": 8,
            "team_size": 1,
            "game_speed": 2,
            "self_play": False,
            "spawn_opponents": False,
            "action_parser": env_id,
            "obs_builder": env_id,
            "terminal_conditions": env_id,
            "reward_function": env_id,
            "state_setter": env_id
          },
    }

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