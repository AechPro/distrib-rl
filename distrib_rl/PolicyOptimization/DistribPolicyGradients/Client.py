from distrib_rl.PolicyOptimization.DistribPolicyGradients import Configurator
from distrib_rl.Distrib import RedisClient, RedisKeys, RedisServer
from distrib_rl.Experience import DistribExperienceManager
import torch
import time
import numpy as np


class Client(object):
    def __init__(self):
        self.strategy_optimizer = None
        self.exp_manager = None
        self.experience = None
        self.value_net = None
        self.client = None
        self.policy = None
        self.agent = None
        self.cfg = None
        self.env = None

        self.last_checked = 0

    def train(self):
        running = True
        while running:
            self.check_server_status()
            self.collect()
            self.update_models()

        self.client.disconnect()

    @torch.no_grad()
    def collect(self):
        t1 = time.perf_counter()
        n_sec = 1
        value_estimator = self.value_net
        exp_manager = self.exp_manager
        agent = self.agent
        client = self.client
        gamma = self.cfg["policy_optimizer"]["gamma"]
        lmbda = self.cfg["policy_optimizer"]["gae_lambda"]

        trajectories = agent.gather_timesteps(self.policy, self.env, num_seconds=n_sec)
        reward_stats = client.get_reward_stats()

        trajectories_to_send = []
        total_timesteps = 0
        for trajectory in trajectories:
            values = value_estimator.get_output(np.asarray(trajectory.obs + [trajectory.final_obs])).flatten().tolist()
            trajectory.finalize(gamma=gamma, reward_stats=reward_stats, values=values, lmbda=lmbda)

            n_timesteps = len(trajectory.rewards)
            trajectories_to_send.append(trajectory.serialize())
            total_timesteps += n_timesteps

        exp_manager.push_trajectories(trajectories_to_send)
        if len(agent.ep_rewards) > 0:
            client.push_data(RedisKeys.CLIENT_POLICY_REWARD_KEY, agent.ep_rewards)
            agent.ep_rewards = []
        print("transmitted {} in {:7.5f}".format(total_timesteps, time.perf_counter() - t1))

    def update_models(self):
        policy_params, value_params, strategy_frames, strategy_history, success = self.client.get_latest_update()
        if not success:
            return False

        self.policy.set_trainable_flat(policy_params)
        self.value_net.set_trainable_flat(value_params)
        self.strategy_optimizer.set_from_server(strategy_frames, strategy_history)

        return True

    def check_server_status(self):
        server_status_flag = self.client.check_server_status()

        if server_status_flag == RedisServer.RESET_STATUS or \
           server_status_flag == None:
            self.reset()

        elif server_status_flag == RedisServer.RECONFIGURE_STATUS  or \
             server_status_flag == RedisServer.INITIALIZING_STATUS or \
             server_status_flag == RedisServer.STOPPING_STATUS:

            self.reconfigure()
        elif server_status_flag == RedisServer.AWAITING_ENV_SPACES_STATUS:
            self.transmit_env_spaces()

    def reset(self):
        print("CLIENT RESETTING")
        if self.env is not None:
            self.env.close()
            self.env = None

        self.reconfigure()

    def reconfigure(self):
        print("CLIENT RECONFIGURING")
        if self.cfg is not None:
            self.cfg.clear()
        if self.client is not None:
            self.client.disconnect()

        self.configure()

    def transmit_env_spaces(self):
        self.client.transmit_env_spaces(self.policy.input_shape, self.policy.output_shape)

    def configure(self):
        print("CLIENT CONFIGURING")
        env = self.env
        self.__init__()

        self.client = RedisClient()

        self.client.connect()
        self.cfg = self.client.get_cfg()

        self.env, self.experience, gradient_builder, policy_gradient_optimizer, value_gradient_optimizer, \
        self.agent, self.policy, self.strategy_optimizer, adaptive_omega, self.value_net, \
        novelty_gradient_optimizer, learner = Configurator.build_vars(self.cfg, existing_env=env)

        self.env.reset()
        self.transmit_env_spaces()
        self.exp_manager = DistribExperienceManager(self.cfg, client=self.client)
        self.last_checked = time.time()

        print("Fetching initial models...")
        while not self.update_models():
            time.sleep(1)

        print("Client configuration complete!")

    def cleanup(self):
        if self.client is not None:
            self.client.disconnect()
