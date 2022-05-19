import os
from distrib_rl.Utils.OmegaSelectors import BayesianBandits

class BayesianOmega(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.omega = cfg["adaptive_omega"]["default"]
        self.min_omega = cfg["adaptive_omega"]["min_value"]
        self.max_omega = cfg["adaptive_omega"]["max_value"]
        n_bins = 40
        arms = [i/n_bins for i in range(0,n_bins+1)]
        # arms = [0, 0.85, 1]
        self.bb = BayesianBandits(arms=arms)

    def step(self, theta_reward):
        self.bb.update_dists(theta_reward)
        self.omega = self.bb.sample()

    def save(self, path):
        with open(os.path.join(path, "omega.dat"), 'w') as f:
            f.write("{}".format(self.omega))

    def cleanup(self):
        pass