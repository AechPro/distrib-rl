from redis import Redis
from distrib_rl.Utils import RedisHelpers as helpers, CompressionSerialisation as cser
from distrib_rl.Distrib import RedisKeys, RedisServer
import time
import pyjson5 as json
import os


class RedisClient(object):
    def __init__(self):
        self.redis = None
        self.current_epoch = -1
        self.max_queue_size = 0

    def connect(self):
        ip = os.environ.get("REDIS_HOST", default='localhost')
        port = os.environ.get("REDIS_PORT", default=6379)
        password = os.environ.get("REDIS_PASSWORD", default=None)

        self.redis = Redis(host=ip, port=port, password=password)

    def push_data(self, key, data, encoded=False):
        red = self.redis
        if not encoded:
            encoded_data = helpers.encode_numpy(data)
        else:
            encoded_data = data

        red.lpush(key, encoded_data)
        red.ltrim(key, 0, self.max_queue_size)

    def get_reward_stats(self):
        mean = self.redis.get(RedisKeys.RUNNING_REWARD_MEAN_KEY)
        std = self.redis.get(RedisKeys.RUNNING_REWARD_STD_KEY)

        if mean is None or std is None:
            mean = 0
            std = 1

        return float(mean), float(std)

    def get_latest_update(self):
        red = self.redis
        epoch = red.get(RedisKeys.SERVER_CURRENT_UPDATE_KEY)
        if epoch is not None:
            epoch = int(epoch)

        if epoch == self.current_epoch or epoch is None:
            return None, None, None, None, False

        self.current_epoch = epoch

        pipe = red.pipeline()
        pipe.get(RedisKeys.SERVER_POLICY_PARAMS_KEY)
        pipe.get(RedisKeys.SERVER_VAL_PARAMS_KEY)
        pipe.get(RedisKeys.SERVER_STRATEGY_FRAMES_KEY)
        pipe.get(RedisKeys.SERVER_STRATEGY_HISTORY_KEY)
        results = pipe.execute()

        encoded_policy, encoded_val, encoded_frames, encoded_history = results

        policy = helpers.decode_numpy(encoded_policy)
        frames = helpers.decode_numpy(encoded_frames)
        hist = helpers.decode_numpy(encoded_history)
        val = helpers.decode_numpy(encoded_val)

        return policy, val, frames, hist, True

    def get_cfg(self):
        import numpy as np
        import random
        import torch

        print("Fetching new config...")
        while True:
            status = self.redis.get(RedisKeys.SERVER_CURRENT_STATUS_KEY)
            if status is not None:
                status = status.decode("utf-8")

            print("Waiting for server to start...",status)

            if status == RedisServer.RUNNING_STATUS or status == RedisServer.AWAITING_ENV_SPACES_STATUS:
                break

            time.sleep(1)
        
        cfg = dict(json.loads(self.redis.get(RedisKeys.SERVER_CONFIG_KEY).decode("utf-8")))
        self.max_queue_size = cfg["experience_replay"]["max_buffer_size"]
        print("Fetched new config!")

        cfg["rng"] = np.random.RandomState(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])

        return cfg

    def check_server_status(self):
        status = self.redis.get(RedisKeys.SERVER_CURRENT_STATUS_KEY)
        if status is None:
            return None

        return status.decode("utf-8")

    def transmit_env_spaces(self, input_shape, output_shape):
        encoded = cser.pack((input_shape, output_shape))
        self.redis.set(RedisKeys.ENV_SPACES_KEY, encoded)


    def disconnect(self):
        self.redis.close()
