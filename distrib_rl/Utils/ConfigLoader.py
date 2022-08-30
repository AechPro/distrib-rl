import pyjson5 as json
import os


def load_config(file_path=None, file_name=None):
    default_path = "resources/configs"
    if file_path is not None:
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE CONFIG FILE IN PATH:\n", file_path, "\n")
            raise FileNotFoundError

    elif file_name is not None:
        file_path = "".join([default_path, "/", file_name])
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE CONFIG FILE IN PATH:\n", file_path, "\n")
            raise FileNotFoundError

    config = dict(json.load(open(file_path, "r")))

    if "device" in config.keys():
        import torch

        if not torch.cuda.is_available():
            config["device"] = "cpu"

    return config


def save_config(file_path, cfg):
    rng = cfg["rng"]
    del cfg["rng"]
    data = json.dumps(cfg)
    with open(file_path, "w") as f:
        f.write(data)
    print("Saved config to {}".format(file_path))
    cfg["rng"] = rng
