from Experience import ExperienceReplay, Timestep, Trajectory
import numpy as np

def run_test():
    num_timesteps = 10
    cfg = {}
    cfg["experience_replay"] = {"max_buffer_size" : num_timesteps}
    cfg["rng"] = np.random.RandomState(12345)

    trajectory = Trajectory()
    replay = ExperienceReplay(cfg)

    for i in range(num_timesteps*2):
        ts = Timestep()
        ts.action = i%2
        ts.log_prob = i//2
        ts.next_obs = i+1
        ts.reward = i*2
        ts.obs = i
        ts.done = False
        print("Timestep {}: {}".format(i, ts.serialize()))
        trajectory.register_timestep(ts)
    trajectory.finalize(gamma=0.95)

    print("\nFinalized trajectory:\n{}\n".format(trajectory.serialize()))
    replay.register_trajectory(trajectory)

    batch = replay.get_batch(num_timesteps//2)
    random_batch = replay.get_random_batch(num_timesteps//2)
    print("Replay batch:\n{}\n",batch)
    print("Replay random batch:\n{}\n",random_batch)

if __name__ == "__main__":
    run_test()