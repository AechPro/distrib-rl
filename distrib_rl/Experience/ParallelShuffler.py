from distrib_rl.MPFramework import MPFProcess


class ParallelShuffler(MPFProcess):
    def __init__(self, name, loop_wait_time=None):
        super().__init__(name, loop_wait_time)
        self.cfg = None
        self.exp_manager = None
        self.server = None
        self.total_ts = 0
        self.ts_per_update = 0
        self.sleep_fn = 0
        self.batch_size = 0
        self.buffer = []

    def init(self):
        import numpy
        from distrib_rl.Experience import DistribExperienceManager
        from distrib_rl.Distrib import RedisServer
        from time import sleep

        self.sleep_fn = sleep

        self.task_checker.wait_for_initialization(header="initialization_data")
        self.cfg = self.task_checker.latest_data.copy()
        self.cfg["rng"] = numpy.random.RandomState(self.cfg["seed"])
        self.server = RedisServer(self.cfg["experience_replay"]["max_buffer_size"])
        self.server.connect(new_server_instance=False)
        self.exp_manager = DistribExperienceManager(self.cfg, server=self.server)

        self.ts_per_update = int(round(self.cfg["policy_optimizer"]["new_returns_proportion"]*self.cfg["experience_replay"]["max_buffer_size"]))
        self.batch_size = self.cfg["policy_optimizer"]["batch_size"]

    def update(self, header, data):
        pass

    def step(self):
        pass

    def publish(self):
        if not self.results_publisher.is_empty():
            self.sleep_fn(0.01)
            return

        publisher = self.results_publisher
        for batch in self.buffer:
            publisher.publish(header="experience_batch", data=batch)

        buffer = []
        ts_collected, fps = self.exp_manager.get_timesteps_as_batches(self.ts_per_update, self.batch_size)

        for batch in self.exp_manager.experience.get_all_batches_shuffled(self.batch_size):
            buffer.append(batch)

        rew_mean = self.exp_manager.experience.reward_stats.mean[0]
        rew_std = self.exp_manager.experience.reward_stats.std[0]
        publisher.publish(header="misc_data", data=(rew_mean, rew_std, ts_collected, fps))
        self.buffer = buffer

    def cleanup(self):
        print("SHUTTING DOWN SHUFFLING PROCESS")
        if self.server is not None:
            self.server.disconnect()

        if self.exp_manager is not None:
            self.exp_manager.cleanup()

        if self.cfg is not None:
            self.cfg.clear()

        print("SHUFFLING PROCESS SHUTDOWN COMPLETE")
