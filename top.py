import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)

from distrib_rl.utils import training_replay, config_loader


def main():
    cfg = config_loader.load_config(file_name="test_config.json")
    cfg["rng"] = np.random.RandomState(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    #opt = PolicyGradients(cfg)
    #opt = SB3Optimizer(cfg)
    #opt.train()

    training_replay.run()


if __name__ == "__main__":
    main()
