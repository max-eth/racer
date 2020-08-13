import pickle

import sacred
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

import torch
import random
import numpy as np
import functools


def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def write_pickle(o, fname):
    with open(fname, "wb") as f:
        pickle.dump(o, f)


def flatten_2d(l):
    return [item for sublist in l for item in sublist]


def flatten_parameters(parameters):
    return np.concatenate([p.flatten() for p in parameters])


def build_parameters(parameter_shapes, parameters_flattened):
    parameters = []
    index = 0
    for shape in parameter_shapes:
        size = functools.reduce(lambda a, b: a * b, shape)
        parameters.append(parameters_flattened[index : index + size].reshape(*shape))
        index += size
    return parameters


def load_pixels(get_dimensions=False):

    pixels = []
    height = 0
    width = None
    with open("resources/use_pixels.txt") as f:
        for i, line in enumerate(f):
            line = line[:-1]  # cut off \n
            height += 1

            for j, c in enumerate(line):
                if c == "1":
                    pixels.append((i, j))

            if width is not None and len(line) != width:
                raise Exception("Different line lengths in use_pixels")
            width = len(line)
    width *= 2
    reflected_pixels = []
    for x, y in pixels:
        reflected_pixels.append((x, 31 - y))

    pixels += reflected_pixels

    for x, y in pixels:
        assert 0 <= x < 32
        assert 0 <= y < 32
    assert len(pixels) == len(set(pixels))

    if get_dimensions:
        return pixels, (width, height)
    else:
        return pixels

def setup_mongo(ex):
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


def setup_sacred_experiment(ex: sacred.Experiment):

    #setup_mongo(ex)

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
