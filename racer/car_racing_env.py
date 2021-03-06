import types

import gym
import numpy as np
import sacred
from sacred import Ingredient
from racer.car_racing_wrapper import CarRacingWrapper


import skimage.transform
import skimage.color

car_racing_env = Ingredient("car_racing_env")

global_env = None


@car_racing_env.config
def config():
    enable_abs = True
    enable_linear_speed = True
    enable_angular_speed = True
    enable_steering = True
    headless = False


@car_racing_env.capture
def image_size():
    image_dim = 32
    return image_dim


@car_racing_env.capture
def feature_size(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
):
    """ The length of the environment features """
    total_length = 0

    if enable_linear_speed:
        total_length += 1

    if enable_angular_speed:
        total_length += 1

    if enable_abs:
        total_length += 4

    if enable_steering:
        total_length += 1

    return total_length


@car_racing_env.capture
def feature_names(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
):
    names = (
        []
    )  # DO NOT CHANGE THE ORDER. it is derived from car_racing_wrappe.get_state

    if enable_linear_speed:
        names.append("linear_speed")

    if enable_angular_speed:
        names.append("angular_speed")

    if enable_abs:
        names += ["wheel_{}_omega".format(i) for i in range(4)]

    if enable_steering:
        names += ["steering"]

    return names


@car_racing_env.capture
def init_env(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
    headless,
    track_data=None,
    render_view=False,
):

    assert not (headless and track_data is None)
    env = CarRacingWrapper(
        enable_linear_speed,
        enable_angular_speed,
        enable_abs,
        enable_steering,
        headless,
        prerendered_data=track_data,
        render_view=render_view,
    )
    global global_env
    global_env = env
    return env


def get_env():
    return global_env


@car_racing_env.capture
def get_track_data(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
    track_data=None,
):

    env = CarRacingWrapper(
        enable_linear_speed, enable_angular_speed, enable_abs, enable_steering
    )
    return env.export()
