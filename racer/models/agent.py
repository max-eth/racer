from multiprocessing.pool import Pool

import numpy as np
from abc import abstractmethod, ABC
import warnings
from racer.models.parallel_eval import eval

from sacred import SETTINGS

# required to make the config shared across processes
SETTINGS.CONFIG.READ_ONLY_CONFIG = False


class Agent(ABC):
    pool = None

    @staticmethod
    def parallel_evaluate(agents):
        """ Evaluate a list of agents on an environment in parallel.

            :param agents: the list of agents
            :return: a list of floats corresponding to the evaluation results of the agents
        """
        if Agent.pool is None:
            print("creating new pool")
            Agent.pool = Pool(processes=4)
        result = Agent.pool.map(eval, agents)
        return result

    @staticmethod
    def race(env, agents, focus_agent):
        """ Race multiple agents together

            :param agents: the list of agents
            :param focus_agent: the index of the agent to focus on
        """
        old_num_cars = env.num_cars
        env.reset(regen_track=False, num_cars=len(agents))
        env.focus_car = focus_agent

        done = False
        neg_reward_count = 0
        while not done:

            # step every agent
            actions = [agent.act(*env.states[i]) for i, agent in enumerate(agents)]
            _, step_reward, done, _ = env.step(*actions)
            if step_reward < 0:
                neg_reward_count += 1
            else:
                neg_reward_count = 0
            if neg_reward_count > 100:
                # stop early
                break
            env.render(mode="human", store_frames=True)
        env.focus_car = 0
        env.reset(regen_track=False, num_cars=old_num_cars)

    @abstractmethod
    def act(self, image, other) -> np.ndarray:
        """ Perform an action. Should return an ndarray ``result`` with shape ``(3,)``, where
            result[0] = steering (range [-1, 1])
            result[1] = gas (range [0, 1])
            result[2] = brake (range [0, 1])
        """
        ...

    def evaluate(self, env, visible=False, store_frames=False, car_id=0) -> float:
        """ Evaluate this agent on the environment, and return its fitness
            :param visible: whether to render the run in a window
        """
        env.reset(regen_track=False)

        done = False
        neg_reward_count = 0
        # progress = tqdm()
        # yappi.set_clock_type("cpu")  # Use set_clock_type("wall") for wall time
        # yappi.start()
        while not done:
            action = self.act(*env.states[car_id])
            _, step_reward, done, _ = env.step(action)
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
        # print(env.max_reward)
        return env.max_reward
