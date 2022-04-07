from Experiments import Experiment, ExperimentLoader
import os

class ExperimentManager(object):
    def __init__(self, optimizer):
        self.optimization_manager = optimizer
        self.experiments = []

    def run_experiments(self):
        for experiment in self.experiments:
            experiment.init()
            while not experiment.step():
                continue

        self.optimization_manager.cleanup()

    def load_experiment(self, filepath):
        experiment_json = ExperimentLoader.load_experiment(file_path=filepath)
        experiment = Experiment(experiment_json, self.optimization_manager)
        self.experiments.append(experiment)

    def load_experiments(self, folder_path):
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            self.load_experiment(filepath)
