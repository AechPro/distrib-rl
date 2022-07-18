from distrib_rl.MPFramework import MPFProcessHandler
from distrib_rl.PolicyOptimization.DistribPolicyGradients import Configurator
from distrib_rl.Distrib import RedisClient, RedisServer
from distrib_rl.PolicyOptimization.DistribPolicyGradients.ClientTrajectoryFinalizer import ClientTrajectoryFinalizer
import torch
import time

class Client(object):
    def __init__(self):
        self.strategy_optimizer = None
        self.experience = None
        self.client = None
        self.policy = None
        self.agent = None
        self.cfg = None
        self.env = None
        self.trajectory_finalizer_handler = None

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
        n_sec = 1
        agent = self.agent

        for trajectory in agent.gather_timesteps(self.policy, self.env, num_seconds=n_sec):
            self.trajectory_finalizer_handler.put(ClientTrajectoryFinalizer.HEADER_TRAJECTORY, data=trajectory)

        self.trajectory_finalizer_handler.put(ClientTrajectoryFinalizer.HEADER_FLUSH,
                                              data=agent.ep_rewards if len(agent.ep_rewards) > 0 else None)
        agent.ep_rewards = []
    
    def update_models(self):
        policy_params, strategy_frames, strategy_history, success = self.client.get_latest_update()
        if not success:
            return False

        self.policy.set_trainable_flat(policy_params)
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
        if self.trajectory_finalizer_handler is not None:
            self.trajectory_finalizer_handler.put(ClientTrajectoryFinalizer.HEADER_FLUSH,
                                                  data=self.agent.ep_rewards if self.agent and len(self.agent.ep_rewards) > 0 else None)
            self.trajectory_finalizer_handler.close()

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
        self.agent, self.policy, self.strategy_optimizer, adaptive_omega, value_net, \
        novelty_gradient_optimizer, learner = Configurator.build_vars(self.cfg, existing_env=env)

        self.env.reset()
        self.transmit_env_spaces()
        self.last_checked = time.time()

        print("Fetching initial models...")
        while not self.update_models():
            time.sleep(1)

        tfh_cfg = self.cfg.copy()
        tfh_cfg["env_space_shapes"] = (self.policy.input_shape, self.policy.output_shape)
        self.trajectory_finalizer_handler = MPFProcessHandler()
        self.trajectory_finalizer_handler.setup_process(ClientTrajectoryFinalizer())
        self.trajectory_finalizer_handler.put(ClientTrajectoryFinalizer.HEADER_RESET, tfh_cfg)

        print("Client configuration complete!")

    def cleanup(self):
        try:
            if self.client is not None:
                self.client.disconnect()
        except:
            pass

        try:
            if self.env is not None:
                self.env.close()
        except:
            pass

        try:
            if self.trajectory_finalizer_handler is not None:
                self.trajectory_finalizer_handler.put(ClientTrajectoryFinalizer.HEADER_FLUSH,
                                                      data=self.agent.ep_rewards if self.agent and len(self.agent.ep_rewards) > 0 else None)
                self.trajectory_finalizer_handler.close()
        except:
            pass
