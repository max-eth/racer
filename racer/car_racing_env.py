import types

import gym
import numpy as np
import sacred
from sacred import Ingredient
from racer.car_racing_wrapper import CarRacingWrapper


import skimage.transform
import skimage.color

car_racing_env = Ingredient("car_racing_env")


@car_racing_env.config
def config():
    enable_abs = True
    enable_linear_speed = True
    enable_angular_speed = True
    enable_steering = True

    # larger values mean more scaling
    image_scaling = 3


@car_racing_env.capture
def image_size(image_scaling):
    image_dim = 96
    image_dim //= image_scaling
    return image_dim


@car_racing_env.capture
def feature_size(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
    image_scaling,
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
def get_env(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
    image_scaling,
):

    env = CarRacingWrapper(enable_linear_speed, enable_angular_speed, enable_abs, enable_steering, image_scaling)
    return env
