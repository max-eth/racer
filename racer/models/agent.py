import numpy as np
from abc import abstractmethod, ABC
import warnings

import yappi
from tqdm import tqdm

from racer.car_racing_env import get_env


class Agent(ABC):
    @abstractmethod
    def act(self, image, other) -> np.ndarray:
        """ Perform an action. Should return an ndarray ``result`` with shape ``(3,)``, where
            result[0] = steering (range [-1, 1])
            result[1] = gas (range [0, 1])
            result[2] = brake (range [0, 1])
        """
        ...

    def evaluate(self, env, visible=False, store_frames=False) -> float:
        """ Evaluate this agent on the environment, and return its fitness
            :param visible: whether to render the run in a window
        """

        if env.t > 0.02:
            warnings.warn("This environment has not been reset")
        done = False
        neg_reward_count = 0
        # progress = tqdm()
        # yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        # yappi.start()
        while not done:
            action = self.act(*env.get_state())
            _, step_reward, done, _ = env.step(action=action)
            if step_reward < 0:
                neg_reward_count += 1
            else:
                neg_reward_count = 0
            if neg_reward_count > 100:
                # print("Stopping early")
                break
            if visible:
                env.render(mode="human", store_frames=store_frames)
            # progress.update()
        # progress.close()
        # yappi.get_func_stats().print_all()
        print(env.max_reward)
        return env.max_reward
