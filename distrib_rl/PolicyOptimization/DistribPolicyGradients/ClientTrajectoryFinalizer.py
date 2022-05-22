import time

import torch
from distrib_rl.Distrib import RedisClient, RedisKeys
from distrib_rl.Distrib.RedisServer import RedisServer
from distrib_rl.MPFramework import MPFProcess
import numpy as np
from distrib_rl.Policies import PolicyFactory


class ClientTrajectoryFinalizer(MPFProcess):
    HEADER_INITIALIZATION="initialization"
    HEADER_FLUSH = "flush_data"
    HEADER_TRAJECTORY = "trajectory"

    def __init__(self, loop_wait_period=0.1):
        super().__init__("ClientTrajectoryFinalizer", loop_wait_period=0.1, process_all_updates=True)
        self.handlers = {
            ClientTrajectoryFinalizer.HEADER_INITIALIZATION: self._init,
            ClientTrajectoryFinalizer.HEADER_FLUSH: self._flush,
            ClientTrajectoryFinalizer.HEADER_TRAJECTORY: self._trajectory
        }
    
    def init(self):
        self.client = None
        self.cfg = None
        self.gamma = None
        self.lmbda = None
        self.value_estimator = None
        self.reward_stats = None

        self.trajectories_to_send = []
        self.total_timesteps = 0
        self.t0 = None
        
    def update(self, header, data):
        handler = self.handlers.get(header, None)
        if not handler:
            print(f"WARNING: ClientTrajectoryFinalizer received unknown message header '{header}'")
            return

        handler(data)

    def _init(self, cfg):
        self.cfg = cfg
        self.gamma = self.cfg["policy_optimizer"]["gamma"]
        self.lmbda = self.cfg["policy_optimizer"]["gae_lambda"]

        pf_cfg = {
            "device": cfg.get("device", "cpu"),
            "value_estimator": cfg["value_estimator"]
        }

        models = PolicyFactory.get_from_cfg(pf_cfg, env_space_shapes=cfg["env_space_shapes"])
        self.value_estimator = models["value_estimator"]

        if self.client is not None:
            try:
                self.client.disconnect()
            except:
                pass

        self.client = RedisClient()
        self.client.connect()

        value_params = None
        while value_params is None:
            value_params = self.client.get_latest_value_params()

        print(f"updated value estimator to version {self.client.current_value_epoch}")

        self.value_estimator.set_trainable_flat(value_params)
        self.reward_stats = self.client.get_reward_stats()

        self.trajectories_to_send = []
        self.total_timesteps = 0
    
    def _flush(self, rewards):
        if self._server_is_running():
            t1 = time.perf_counter()

            if len(self.trajectories_to_send) > 0:
                self.client.push_data(RedisKeys.CLIENT_EXPERIENCE_KEY, self.trajectories_to_send)

                t2 = time.perf_counter()

                # we use t1 here because we don't count the time involved in the
                # send, as it's not blocking the game
                seconds = t1 - self.t0
                steps_per_second = float(self.total_timesteps) / seconds

                if rewards:
                    self.client.push_data(RedisKeys.CLIENT_POLICY_REWARD_KEY, rewards)

                print("packed and pushed {} trajectories containing {} timesteps in {:7.5f}s ({:.2f} sps)".format(
                    len(self.trajectories_to_send),
                    self.total_timesteps,
                    t2-t1,
                    steps_per_second))
            else:
                print("No trajectories to send.")

            value_params = self.client.get_latest_value_params()
            if value_params is not None:
                self.value_estimator.set_trainable_flat(value_params)
                print(f"updated value estimator to version {self.client.current_value_epoch}")

            self.reward_stats = self.client.get_reward_stats()

        self.total_timesteps = 0
        self.trajectories_to_send = []
        self.t0 = time.perf_counter()
        print("")
    
    @torch.no_grad()
    def _trajectory(self, trajectory):
        if self.t0 is None:
            self.t0 = time.perf_counter()

        values = self.value_estimator.get_output(np.asarray(trajectory.obs + [trajectory.final_obs])).flatten().tolist()
        trajectory.finalize(gamma=self.gamma,
                            reward_stats=self.reward_stats,
                            values=values,
                            lmbda=self.lmbda)

        n_timesteps = len(trajectory.rewards)
        self.trajectories_to_send.append(trajectory.serialize())
        self.total_timesteps += n_timesteps

    def _server_is_running(self):
        server_status_flag = self.client.check_server_status()
        return server_status_flag == RedisServer.RUNNING_STATUS

    def step(self):
        pass

    def publish(self):
        pass

    def cleanup(self):
        try:
            self._flush()
        except:
            pass

        try:
            self.client.disconnect()
        except:
            pass
