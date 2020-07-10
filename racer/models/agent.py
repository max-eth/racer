import numpy as np
from abc import abstractmethod, ABC

from tqdm import tqdm

from racer.car_racing_env import get_env


class Agent(ABC):
    @abstractmethod
    def parameters(self):
        """ Return all parameters as numpy arrays """
        ...

    @abstractmethod
    def set_parameters(self, parameters):
        """Set parameters from numpy arrays"""
        ...

    @abstractmethod
    def act(self, image, other) -> np.ndarray:
        """ Perform an action. Should return an ndarray ``result`` with shape ``(3,)``, where
            result[0] = steering (range [-1, 1])
            result[1] = gas (range [0, 1])
            result[2] = brake (range [0, 1])
        """
        ...

    def evaluate(self, visible=False) -> float:
        """ Evaluate this agent on the environment, and return its fitness
            :param visible: whether to render the run in a window
        """

        env = get_env()
        env.reset()
        env.step(action=None)
        done = False
        neg_reward_count = 0
        progress = tqdm()
        while not done:
            action = self.act(*env.get_state())
            _, step_reward, done, _ = env.step(action=action)
            if step_reward < 0:
                neg_reward_count += 1
            else:
                neg_reward_count = 0
            if neg_reward_count > 100:
                print("Stopping early")
                break
            #print("{:.4f}\t {:.4f}".format(env.reward, step_reward))
            if visible:
                env.render(mode='human')
            progress.update()
        env.viewer.close()
        return env.reward



