from distrib_rl.Utils import ConfigLoader
from distrib_rl.Experiments.ConfigAdjusters import AdjusterFactory
import os
import numpy as np

class Experiment(object):
    def __init__(self, experiment_json, optimization_manager):
        self.experiment_json = experiment_json

        self.config_adjusters = None
        self.optimization_manager = optimization_manager
        self.cfg = None

        self.current_adjuster_index = 0

        self.num_trials = experiment_json["num_trials_per_adjustment"]
        self.terminal_conditions = experiment_json["terminal_conditions"]

        self.current_trial = 0

        self.base_dir = os.path.join(os.getcwd(), "data", "experiments")
        self.experiment_name = experiment_json["experiment_name"]
        self.adjustment_dir = ""

        self.step_num = 0

    def init(self):
        self.cfg = ConfigLoader.load_config(file_name=self.experiment_json["config_file"])

        self.config_adjusters = AdjusterFactory. \
            build_adjusters_for_experiment(self.experiment_json["config_adjustments"], self.cfg)

        self.config_adjusters[self.current_adjuster_index].reset_config(self.cfg)
        self.config_adjusters[self.current_adjuster_index].adjust_config(self.cfg)
        self.adjustment_dir = self.config_adjusters[self.current_adjuster_index].get_name()
        self.start_trial()

    def step(self):
        if self.is_done():
            return True

        self.optimization_manager.step()
        self.step_num += 1

        if self.step_num % self.experiment_json["steps_per_save"] == 0:
            self.optimization_manager.save_progress()

        if self.optimization_manager.is_done():
            self.step_num = 0
            self.current_trial += 1
            self.next_trial()

        return False

    def get_next_adjustment(self):
        if self.is_done():
            return

        idx = self.current_adjuster_index
        done = self.config_adjusters[idx].is_done()

        if done:
            self.config_adjusters[idx].reset_config(self.cfg)
            self.current_adjuster_index += 1
            idx = self.current_adjuster_index

        if idx >= len(self.config_adjusters):
            return

        self.config_adjusters[idx].step()
        self.config_adjusters[idx].adjust_config(self.cfg)
        self.adjustment_dir = self.config_adjusters[idx].get_name()

    def next_trial(self):
        if self.current_trial >= self.num_trials:
            self.current_trial = 0
            self.get_next_adjustment()

            if self.is_done():
                print("Experiment complete!")
                self.optimization_manager.reset()
                return

            elif self.config_adjusters[self.current_adjuster_index].reset_per_increment():
                self.optimization_manager.reset()

            else:
                self.optimization_manager.reconfigure()

        else:
            self.optimization_manager.reconfigure()

        print("Starting new trial...")
        self.start_trial()

    def start_trial(self):
        current_trial_dir = os.path.join(self.base_dir, self.experiment_json["experiment_name"],
                                                 self.adjustment_dir, str(self.current_trial))

        experiment_name = "{}-{}-{}".format(self.experiment_json["experiment_name"],
                                            self.adjustment_dir, self.current_trial)

        self.cfg["experiment_name"] = experiment_name

        self.cfg["seed"] += 1
        self.cfg["rng"] = np.random.RandomState(int(self.cfg["seed"]))

        self.optimization_manager.configure(self.cfg)
        self.optimization_manager.set_base_dir(current_trial_dir)
        self.optimization_manager.set_terminal_conditions(self.terminal_conditions)
        print("Trial started!")

    def is_done(self):
        return self.current_adjuster_index >= len(self.config_adjusters)