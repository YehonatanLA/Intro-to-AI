import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random

# TODO: test time limit
TIME_THRESHOLD = 0.99


# TODO: section a : 3
def check_time_limit(start, time_limit):
    if time.time() - start < time_limit * TIME_THRESHOLD:
        return True
    else:
        return False


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    pass


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def check_agent_time(self, start, time_limit):
        return check_time_limit(start, time_limit)

    # TODO: section b : 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        # TODO: change depth to time limit
        start = time.time()
        return self.minimax(env, agent_id, time_limit, agent_id, start)

    def minimax(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started):
        if self.check_agent_time(time_started, time_limit) or env.done():
            return self.heuristic(env, agent_id)

        if agent_id == original_agent_id:
            return self.max_value(env, agent_id, time_limit, original_agent_id, time_started)
        else:
            return self.min_value(env, agent_id, time_limit, original_agent_id, time_started)

    def max_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started):
        v = float('-inf')
        for op, child in self.successors(env, agent_id):
            v = max(v, self.minimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started))
        return v

    def min_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started):
        v = float('inf')
        for op, child in self.successors(env, agent_id):
            v = min(v, self.minimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started))
        return v

    def successors(self, env: WarehouseEnv, robot_id: int):
        operators = env.get_legal_operators(robot_id)
        children = [env.clone() for _ in operators]
        for child, op in zip(children, operators):
            child.apply_operator(robot_id, op)
        return operators, children


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move north", "move east", "move north", "move north", "pick_up", "move east", "move east",
                           "move south", "move south", "move south", "move south", "drop_off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)
