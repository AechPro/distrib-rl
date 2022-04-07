import json
import os


def load_experiment(file_path):
    if file_path is not None:
        if not os.path.exists(file_path):
            print("\nUNABLE TO LOCATE EXPERIMENT FILE IN PATH:\n",file_path,"\n")
            raise FileNotFoundError

    experiment = dict(json.load(open(file_path, 'r')))
    return experiment