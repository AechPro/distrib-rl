from distrib_rl.MPFramework import MPFProcessHandler
from distrib_rl.Experience import ParallelShuffler
import time

class ParallelExperienceManager(object):
    def __init__(self, cfg):
        self.process_handler = None
        self.cfg = cfg
        self.rew_mean = 0
        self.rew_std = 1
        self.ts_collected = 0
        self.steps_per_second = 0
        self._init_process()

    def get_all_batches_shuffled(self):
        self.ts_collected = 0
        new_batches = []
        handler = self.process_handler
        while len(new_batches) == 0:
            batches = handler.get_all()
            if batches is None:
                time.sleep(0.01)
                continue

            for data in batches:
                header, msg = data
                if header == "misc_data":
                    self.rew_mean, self.rew_std, ts_collected, self.steps_per_second =  msg
                    self.ts_collected += ts_collected
                else:
                    new_batches.append(msg)
        return new_batches

    def _init_process(self):
        rng = self.cfg["rng"]
        del self.cfg["rng"]
        handler = MPFProcessHandler()
        process = ParallelShuffler("server_data_shuffling_process")
        handler.setup_process(process)
        handler.put(header="initialization_data", data=self.cfg)
        self.process_handler = handler
        self.cfg["rng"] = rng

    def cleanup(self):
        self.process_handler.stop()
        self.cfg = None