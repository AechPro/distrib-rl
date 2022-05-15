from distrib_rl.PolicyOptimization.DistribPolicyGradients import Configurator
from distrib_rl.MARL import OpponentSelector
from distrib_rl.Distrib import RedisServer, RedisKeys, CompressionSerialisation as cser
from distrib_rl.Utils import ConfigLoader
from distrib_rl.Experience import ParallelExperienceManager
import time
import numpy as np
import wandb
import os


class Server(object):
    def __init__(self):
        self.novelty_gradient_optimizer = None
        self.policy_gradient_optimizer = None
        self.value_gradient_optimizer = None
        self.strategy_optimizer = None
        self.opponent_selector = None
        self.gradient_builder = None
        self.adaptive_omega = None
        self.policy_reward = None
        self.exp_manager = None
        self.experience = None
        self.wandb_run = None
        self.value_net = None
        self.learner = None
        self.policy = None
        self.server = None
        self.agent = None
        self.cfg = None

        self.cumulative_ts = 0
        self.epoch_info = {}
        self.base_directory = ""
        self.terminal_conditions = {}
        self.epoch = 0


    def train(self):
        while not self.is_done():
            self.step()

    def step(self):
        t1 = time.time()
        self.update()
        self.opponent_selector.submit_policy(self.policy.get_trainable_flat())
        self.strategy_optimizer.update()
        self.update_server(self.epoch)

        self.get_policy_reward()
        if self.epoch_info["ts_consumed"] > 0:
            self.adaptive_omega.step(self.policy_reward)

        self.epoch_info["epoch_time"] = time.time() - t1
        self.epoch_info["epoch"] = self.epoch

        self.epoch_info["mean_policy_reward"] = self.policy_reward

        self.epoch += 1

        if self.epoch_info["ts_consumed"] > 0:
            self.server.redis.set(RedisKeys.SERVER_CUMULATIVE_TIMESTEPS_KEY, self.cumulative_ts)
            self.server.redis.set(RedisKeys.RUNNING_REWARD_MEAN_KEY, float(self.exp_manager.rew_mean))
            self.server.redis.set(RedisKeys.RUNNING_REWARD_STD_KEY, float(self.exp_manager.rew_std))
            self.epoch_info["steps_per_second"] = int(round(self.exp_manager.steps_per_second))
            self.report_epoch()

        self.epoch_info.clear()

    def update(self):
        before = self.policy.get_trainable_flat(force_update=True)

        self.update_models()
        self.update_policy_novelty()

        after = self.policy.get_trainable_flat(force_update=True)
        self.epoch_info["update_magnitude"] = np.linalg.norm(before - after)

    def update_models(self):
        self.epoch_info["learner"] = self.learner.learn(self.exp_manager)
        ts_collected = self.exp_manager.ts_collected

        self.epoch_info["val_loss"] = self.epoch_info["learner"]["val_loss"]
        del self.epoch_info["learner"]["val_loss"]

        self.epoch_info["ts_consumed"] = ts_collected

        self.cumulative_ts += ts_collected
        self.epoch_info["cumulative_timesteps"] = self.cumulative_ts

    def update_policy_novelty(self):
        w = self.adaptive_omega.omega
        self.epoch_info["policy_novelty"] = self.strategy_optimizer.compute_policy_novelty()
        self.epoch_info["omega"] = w

    def get_policy_reward(self):
        rewards = self.server.get_policy_rewards()
        if len(rewards) == 0:
            return

        reward = self.policy_reward
        if reward is None:
            reward = np.mean(rewards)
        else:
            alpha = 0.99
            beta = 1 - alpha
            for rew in rewards:
                reward = reward * alpha + rew * beta
        # print("Collected {} policy rewards\nStatistics: {}".format(len(rewards), RLMath.compute_array_stats(rewards)))

        self.policy_reward = reward
        self.server.redis.set(RedisKeys.MEAN_POLICY_REWARD_KEY, reward)


    def update_server(self, current_epoch):
        policy_params = self.policy.get_trainable_flat(force_update=True)
        value_params = self.value_net.get_trainable_flat(force_update=True)
        strategy_frames, strategy_history = self.strategy_optimizer.serialize()

        self.server.push_update(policy_params, value_params, strategy_frames, strategy_history, current_epoch)

    def save_progress(self):
        print("SAVING TO BASE",self.base_directory)

        cfg_path = os.path.join(self.base_directory, "config.json")
        if not os.path.exists(cfg_path):
            ConfigLoader.save_config(cfg_path, self.cfg)

        models_dir = os.path.join(self.base_directory, "models")
        optim_dir = os.path.join(self.base_directory, "grad_optimizer")

        self.policy.save(models_dir, "policy_{}".format(self.epoch))
        self.value_net.save(models_dir, "value_net_{}".format(self.epoch))
        self.policy_gradient_optimizer.save(optim_dir, "policy_gradient_optimizer_{}".format(self.epoch))
        self.value_gradient_optimizer.save(optim_dir, "value_gradient_optimizer_{}".format(self.epoch))

    def load_weights(self, load_dir, load_epoch):
        print(f"LOADING WEIGHTS FROM EPOCH {load_epoch} OF RUN AT PATH {load_dir}")
        models_dir = os.path.join(load_dir, "models")
        optim_dir = os.path.join(load_dir, "grad_optimizer")

        self.policy.load(models_dir, "policy_{}.npy".format(load_epoch))
        self.value_net.load(models_dir, "value_net_{}.npy".format(load_epoch))
        self.policy_gradient_optimizer.load(optim_dir, "policy_gradient_optimizer_{}".format(load_epoch))
        self.value_gradient_optimizer.load(optim_dir, "value_gradient_optimizer_{}".format(load_epoch))

    def set_base_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        print("SETTING BASE DIR",directory)
        self.base_directory = directory

    def set_terminal_conditions(self, conditions):
        self.terminal_conditions = conditions

    def is_done(self):
        if self.terminal_conditions["max_epoch"] > 0:
            if self.epoch >= self.terminal_conditions["max_epoch"]:
                return True

        if self.terminal_conditions["max_timesteps"] > 0:
            if self.cumulative_ts >= self.terminal_conditions["max_timesteps"]:
                return True

        if self.terminal_conditions["policy_reward"] > 0:
            if self.policy_reward >= self.terminal_conditions["policy_reward"]:
                return True

        return False

    def setup_redis(self, cfg):
        self.server = RedisServer(cfg["experience_replay"]["max_buffer_size"])
        self.server.connect(clear_existing=True)
        self.server.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.RECONFIGURE_STATUS)
        print("Connected to redis. Waiting for clients...")
        time.sleep(2)
        self.server.push_cfg(cfg)
        self.server.redis.set(RedisKeys.RUNNING_REWARD_MEAN_KEY, 0)
        self.server.redis.set(RedisKeys.RUNNING_REWARD_STD_KEY, 1)
        return self.server.get_env_spaces()


    def reset(self):
        print("SERVER RESETTING")
        self.server.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.RESET_STATUS)
        print("WAITING 2 SECONDS FOR ALL CLIENTS TO CATCH UP ON RESET SIGNAL")
        time.sleep(2)
        self.cleanup()

    def reconfigure(self):
        print("SERVER RECONFIGURING")
        self.server.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.RECONFIGURE_STATUS)
        print("WAITING 2 SECONDS FOR ALL CLIENTS TO CATCH UP ON RECONFIGURE SIGNAL")
        time.sleep(2)
        self.cleanup()

    def configure(self, cfg):
        if cfg["log_to_wandb"]:
            exp_name = cfg["experiment_name"]
            experiment, adjustment, trial = exp_name.split("-")
            self.wandb_run = wandb.init(project=experiment,
                                        group=adjustment,
                                        name="trial {}".format(trial),
                                        config=cfg,
                                        reinit=True)

        terminal_conditions = self.terminal_conditions.copy()
        self.__init__()
        self.cfg = cfg
        self.terminal_conditions = terminal_conditions

        networking_cfg = self.cfg.get("networking", dict(compression="none"))
        compression = networking_cfg.get("compression", "none")
        if compression == "lz4":
            cser.set_compression(cser.LZ4)
        elif compression != "none":
            raise ValueError("Unknown compression type: {}".format(compression))

        in_shape, out_shape = self.setup_redis(cfg)
        print("Server configuring...")
        env, \
        self.experience, \
        self.gradient_builder, \
        self.policy_gradient_optimizer, \
        self.value_gradient_optimizer, \
        self.agent, \
        self.policy, \
        self.strategy_optimizer, \
        self.adaptive_omega, \
        self.value_net, \
        self.novelty_gradient_optimizer, \
        self.learner = Configurator.build_vars(cfg, env_space_shapes=(in_shape, out_shape))

        print("Starting new experience manager...")
        self.exp_manager = ParallelExperienceManager(cfg)
        self.opponent_selector = OpponentSelector(cfg)
        print("Local variables set up")

        self.epoch = 0
        self.cumulative_ts = 0
        self.policy_reward = None
        self.epoch_info = {}

        if "load_prev_weights" in cfg.keys():
            lpw_cfg = cfg["load_prev_weights"]
            if ("path" in lpw_cfg.keys()) and ("epoch" in lpw_cfg.keys()):
                self.load_weights(lpw_cfg["path"], lpw_cfg["epoch"])

        self.opponent_selector.submit_policy(self.policy.get_trainable_flat(force_update=True))
        self.strategy_optimizer.update()
        self.update_server(self.epoch)


        print("sending ready signal...")
        self.server.signal_ready()
        print("server ready!")

        self.epoch += 1

    def report_epoch(self):
        info = self.epoch_info
        if info["mean_policy_reward"] is None:
            return

        if self.cfg["log_to_wandb"]:
            wandb.log(info)

        asterisks = "*"*8
        report = "\n{} BEGIN EPOCH {} REPORT {}\n"\
        "Policy Reward:         {:7.5f}\n"\
        "Policy Novelty:        {:7.5f}\n"\
        "Policy Entropy:        {:7.5f}\n"\
        "Policy Updates:        {:7}\n\n"\
        "KL Divergence:         {:7.5f}\n"\
        "Clip Fraction:         {:7.5f}\n"\
        "Update Magnitude:      {:7.5f}\n"\
        "Omega:                 {:7.5f}\n\n"\
        "TS This Epoch          {:7}\n"\
        "Cumulative TS          {:7}\n"\
        "Steps Per Second       {:7}\n"\
        "Value loss:            {:7.5f}\n" \
        "Learning Rate:         {:7.5f}\n"\
        "Epoch Time:            {:7.5f}\n"\
        "{} END EPOCH {} REPORT {}\n".format(
            asterisks,
            info["epoch"],
            asterisks,
            info["mean_policy_reward"],
            info["policy_novelty"],
            info["learner"]["mean_entropy"],
            info["learner"]["n_updates"],
            info["learner"]["mean_kl"],
            info["learner"]["clip_fraction"],
            info["update_magnitude"],
            info["omega"],
            info["ts_consumed"],
            info["cumulative_timesteps"],
            info["steps_per_second"],
            info["val_loss"],
            info["learner"]["learning_rate"],
            info["epoch_time"],
            asterisks,
            info["epoch"],
            asterisks
        )
        print(report)

    def cleanup(self):
        if self.server is not None:
            self.server.disconnect()

        if self.exp_manager is not None:
            self.exp_manager.cleanup()

        if self.wandb_run is not None:
            self.wandb_run.finish(quiet=True)
