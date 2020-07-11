import numpy as np
from sacred import Ingredient
from racer.car_racing_env import car_racing_env, image_size
from racer.models.agent import Agent

genetic = Ingredient("genetic", ingredients=[car_racing_env])


@genetic.config
def genetic_config():

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


@genetic.capture
def image_feature_size(pixels):
    return len(pixels)


@genetic.capture
def check_pixel_map_size(width, height):
    if width != image_size():
        raise Exception(
            "Use pixels has invalid dimensions. Expected width {} but got {}".format(
                image_size(), width
            )
        )
    if height != image_size():
        raise Exception(
            "Use pixels has invalid dimensions. Expected height {} but got {}".format(
                image_size(), height
            )
        )


class GeneticAgent(Agent):
    @genetic.capture
    def __init__(self, policy_function, pixels):
        check_pixel_map_size()
        self.policy_function = policy_function
        self.pixels = pixels

    def parameters(self):
        return []

    def act(self, image, other) -> np.ndarray:

        if image.shape[0] != 1 or image.shape[1] != 1:
            raise ValueError("Invalid dimensions")

        image_in = [image[0, 0, coords[0], coords[1]] for coords in self.pixels]
        x = image_in + list(other)

        fct_out = self.policy_function(*zip(x, x))
        steering, acceleration = np.tanh(fct_out[0]), np.tanh(fct_out[1])

        gas, brake = max(0, acceleration), max(0, -acceleration)

        return np.array([steering, gas, brake])
