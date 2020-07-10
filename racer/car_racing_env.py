import types

import gym
import numpy as np
import sacred
from sacred import Ingredient


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

    if image_scaling != 1:
        image_dim = 96
        image_dim //= image_scaling
        total_length += image_dim * image_dim

    return total_length


@car_racing_env.capture
def get_env(
    enable_abs,
    enable_linear_speed,
    enable_angular_speed,
    enable_steering,
    image_scaling,
):
    """ Get the environment.

        We monkey-patch in a `get_state` method that returns our state vector.
    """

    env = gym.make("CarRacing-v0")

    def get_state(self):
        vectors = []
        if enable_linear_speed:
            linear_speed = np.sqrt(
                np.square(self.car.hull.linearVelocity[0])
                + np.square(self.car.hull.linearVelocity[1])
            )
            vectors.append(np.array([linear_speed]))

        if enable_angular_speed:
            vectors.append(np.array([self.car.hull.angularVelocity]))

        if enable_abs:
            vectors.append(np.array([self.car.wheels[i].omega for i in range(4)]))

        if enable_steering:
            vectors.append(np.array([self.car.wheels[0].joint.angle]))

        image = skimage.color.rgb2gray(self.state)
        if image_scaling != 1:
            image = skimage.transform.downscale_local_mean(
                image, (image_scaling, image_scaling)
            )

        vectors.append(image.flatten())
        return np.concatenate(vectors, axis=0)

    env.get_state = types.MethodType(get_state, env)
    return env
