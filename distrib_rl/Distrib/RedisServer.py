from redis import Redis
from distrib_rl.Distrib import RedisKeys
from distrib_rl.Distrib.MessageSerialization import MessageSerializer
import time
import pyjson5 as json
import os


class RedisServer(object):
    INITIALIZING_STATUS        = "REDIS_SERVER_INITIALIZING_STATUS"
    RUNNING_STATUS             = "REDIS_SERVER_RUNNING_STATUS"
    STOPPING_STATUS            = "REDIS_SERVER_STOPPING_STATUS"
    RESET_STATUS               = "REDIS_SERVER_RESET_STATUS"
    RECONFIGURE_STATUS         = "REDIS_SERVER_RECONFIGURE_STATUS"
    AWAITING_ENV_SPACES_STATUS = "REDIS_SERVER_AWAITING_ENV_SPACES_STATUS"

    def __init__(self, max_queue_size):
        self.redis = None
        self.max_queue_size = max_queue_size
        self.internal_buffer = []
        self.available_timesteps = 0

        self.last_sps_measure = time.time()
        self.accumulated_sps = 0
        self.steps_per_second = 0
        self._message_serializer = MessageSerializer()

    def connect(self, clear_existing=False, new_server_instance=True):
        ip = os.environ.get("REDIS_HOST", default='localhost')
        port = os.environ.get("REDIS_PORT", default=6379)
        password = os.environ.get("REDIS_PASSWORD", default=None)
        self.redis = Redis(host=ip, port=port, password=password)
        if clear_existing:
            self.redis.flushall()

        if new_server_instance:
            self.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.INITIALIZING_STATUS)
            self.redis.set(RedisKeys.NEW_DATA_AMOUNT_KEY, 0)

    def get_n_timesteps(self, n):
        self._update_buffer()
        while self.available_timesteps < n:
            self._update_buffer()
            time.sleep(0.01)

        n_collected = 0
        buffer = self.internal_buffer
        returns = []

        while n_collected < n:
            ret = buffer.pop(-1)
            trajectory, num_timesteps = ret
            returns.append(trajectory)
            n_collected += num_timesteps

        self.available_timesteps -= n_collected
        return returns

    def get_up_to_n_timesteps(self, n):
        self._update_buffer()
        if self.available_timesteps == 0:
            return []

        n_collected = 0
        buffer = self.internal_buffer
        returns = []

        while len(buffer) > 0 and n_collected < n:
            ret = buffer.pop(-1)
            trajectory, num_timesteps = ret
            returns.append(trajectory)
            n_collected += num_timesteps

        self.available_timesteps -= n_collected
        return returns

    def get_policy_rewards(self):
        # rewards are pushed as packed/compressed lists of scalar values
        # atomic_pop_all returns all entries for a given key as a list, giving
        # us a list of lists of reward scalars. We need to flatten it before we
        # return.
        reward_lists = self._atomic_pop_all(RedisKeys.CLIENT_POLICY_REWARD_KEY)

        # Actual flattening happens here. Not very readable, but it's supposedly
        # the fastest way to flatten a list[list[Any]], per
        # https://stackoverflow.com/a/952952
        return [reward for reward_list in reward_lists for reward in reward_list] 

    def push_update(self, policy_params, val_params, strategy_frames, strategy_history, current_epoch):
        red = self.redis

        packed_policy = self._message_serializer.pack(policy_params)
        packed_val = self._message_serializer.pack(val_params)
        packed_frames = self._message_serializer.pack(strategy_frames)
        packed_history = self._message_serializer.pack(strategy_history)

        pipe = red.pipeline()
        pipe.set(RedisKeys.SERVER_POLICY_PARAMS_KEY, packed_policy)
        pipe.set(RedisKeys.SERVER_VAL_PARAMS_KEY, packed_val)
        pipe.set(RedisKeys.SERVER_STRATEGY_FRAMES_KEY, packed_frames)
        pipe.set(RedisKeys.SERVER_STRATEGY_HISTORY_KEY, packed_history)
        pipe.set(RedisKeys.SERVER_CURRENT_UPDATE_KEY, current_epoch)
        pipe.execute()

    def push_cfg(self, cfg):
        self._configure_serialization(cfg)

        dev = cfg["device"]
        rng = cfg["rng"]

        del cfg["rng"]
        cfg["device"] = "cpu"
        self.redis.set(RedisKeys.SERVER_CONFIG_KEY, json.dumps(cfg))

        cfg["rng"] = rng
        cfg["device"] = dev

    def _configure_serialization(self, cfg):
        networking_cfg = cfg.get("networking", {})
        compression_type = networking_cfg.get("compression", None)
        if compression_type:
            self._message_serializer = MessageSerializer(compression_type=compression_type)
        else:
            self._message_serializer = MessageSerializer()

    def signal_ready(self):
        self.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.RUNNING_STATUS)

        pipe = self.redis.pipeline()
        pipe.delete(RedisKeys.CLIENT_EXPERIENCE_KEY)
        pipe.delete(RedisKeys.CLIENT_POLICY_REWARD_KEY)
        pipe.set(RedisKeys.NEW_DATA_AMOUNT_KEY, 0)

        # Short sleep to let any pre-connected clients update their policies before we erase the existing data.
        time.sleep(1)
        pipe.execute()

    def get_env_spaces(self):
        self.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.AWAITING_ENV_SPACES_STATUS)
        in_space, out_space = None, None

        while in_space is None or out_space is None:
            data = self.redis.get(RedisKeys.ENV_SPACES_KEY)
            if data is None:
                time.sleep(0.1)
                continue

            in_space, out_space = self._message_serializer.unpack(data)
        return in_space, out_space

    def _update_buffer(self):
        returns = self._atomic_pop_all(RedisKeys.CLIENT_EXPERIENCE_KEY)
        collected_timesteps = 0
        for trajectories in returns:
            for serialized_trajectory in trajectories:
                n_timesteps = len(serialized_trajectory[0])
                collected_timesteps += n_timesteps
                self.internal_buffer.append((serialized_trajectory, n_timesteps))
        self.available_timesteps += collected_timesteps

        self._update_sps(collected_timesteps)
        self._trim_buffer()

    def _trim_buffer(self):
        while self.max_queue_size < self.available_timesteps:
            ret = self.internal_buffer.pop(0)
            self.available_timesteps -= ret[1]
            del ret

    def _update_sps(self, collected_timesteps):
        self.accumulated_sps += collected_timesteps
        elapsed = time.time() - self.last_sps_measure

        if elapsed >= 1:
            self.steps_per_second = 0.9 * self.steps_per_second + 0.1 * self.accumulated_sps / elapsed
            self.accumulated_sps = 0
            self.last_sps_measure = time.time()

    def _atomic_pop_all(self, key):
        pipe = self.redis.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.delete(key)
        packed_results = pipe.execute()[0]
        if packed_results is None:
            return []
        return [self._message_serializer.unpack(packed_result) for packed_result in packed_results]

    def disconnect(self):
        if self.redis is not None:
            self.redis.flushall()
            print("\nATTEMPTING TO SET REDIS TO STOPPING STATUS")
            self.redis.set(RedisKeys.SERVER_CURRENT_STATUS_KEY, RedisServer.STOPPING_STATUS)
            self.redis.close()

        del self.internal_buffer
        self.internal_buffer = []
