import pickle

import sacred
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import torch
import random
import numpy as np


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def write_pickle(o, fname):
    with open(fname, "wb") as f:
        pickle.dump(o, f)


def flatten_2d(l):
    return [item for sublist in l for item in sublist]


def setup_sacred_experiment(ex: sacred.Experiment):
    try:
        with open("credentials.txt") as f:
            content = [line.strip() for line in f]
        ip, db_name, account, pw = content
    except Exception as e:
        raise ValueError("Invalid credentials.txt") from e

    ex.observers.append(
        MongoObserver(
            url="mongodb://{}:{}@{}/{}?authMechanism=SCRAM-SHA-1".format(
                account, pw, ip, db_name
            ),
            db_name=db_name,
        )
    )

    ex.captured_out_filter = apply_backspaces_and_linefeeds

    @ex.capture
    def set_seed(_seed):
        random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

    ex.pre_run_hook(set_seed)


def dict_to_device(d, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in d.items()}
