import time

from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance, Robot
import random
import numpy as np

# TODO: test time limit
TIME_THRESHOLD = 0.97
SPECIAL_OPS = ['move east', 'pick up']


# TODO: section a : 3
def check_time_limit(start, time_limit):
    if time.time() - start < time_limit * TIME_THRESHOLD:
        return True
    else:
        return False


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    h_val = 40 + 1000 * env.get_robot(robot_id).credit
    max_charge = 40
    # charge_rate = env.get_robot(robot_id).battery / max_charge
    charge_rate = 1
    current_packages = [p for p in env.packages if p.on_board]
    if not env.get_robot(robot_id).package:
        if len(current_packages) == 1:
            h_val -= charge_rate * manhattan_distance(env.get_robot(robot_id).position, current_packages[0].position)
        else:
            min_dist = charge_rate * min(
                manhattan_distance(env.get_robot(robot_id).position, current_packages[0].position),
                manhattan_distance(env.get_robot(robot_id).position, current_packages[1].position))
            h_val -= min_dist
            # h_val += (1 - charge_rate) * min(
            #     manhattan_distance(env.get_robot(robot_id).position, env.charge_stations[0].position),
            #     manhattan_distance(env.get_robot(robot_id).position, env.charge_stations[1].position))
    else:
        h_val += 100
        dist = charge_rate * manhattan_distance(env.get_robot(robot_id).position,
                                                env.get_robot(robot_id).package.destination)
        h_val -= dist
        # h_val += (1 - charge_rate) * min(manhattan_distance(env.get_robot(robot_id).position, env.charge_stations[0]),
        #                                  manhattan_distance(env.get_robot(robot_id).position, env.charge_stations[1]))
    return h_val


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    @staticmethod
    def got_time(start, time_limit):
        return check_time_limit(start, time_limit)

    # TODO: section b : 1

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 1
        operation = ''
        # helps for printing the children
        # print("\n\n")

        # doing one step of minimax since I want to return the operator
        # the minmax function will return the max value of each child
        try:
            while True:
                # operators = env.get_legal_operators(agent_id)
                ops_children = self.successors(env, agent_id)
                children_values = []
                max_heuristic = float('-inf')

                for op, child in zip(ops_children[0], ops_children[1]):
                    curr_value = self.minimax(child, (agent_id + 1) % 2, time_limit, agent_id, start, depth)
                    children_values.append((curr_value, op))
                    max_heuristic = max(max_heuristic, curr_value)

                # print the children heuristics and the operator
                # operators_and_values = [(j, operators[i]) for i, j in enumerate(children_values)]
                # print("the operators and their values for minimax agent:")
                # print(children_values)

                max_values_operations = [tup[1] for tup in children_values if tup[0] == max_heuristic]
                operation = random.choice(max_values_operations)
                if max_heuristic == float('inf'):
                    raise Exception
                depth += 1

        except:
            # helps for printing the children
            # print("\n\n")
            return operation

    def minimax(self, env: WarehouseEnv, agent_turn, time_limit, original_agent_id, time_started, depth):
        if not self.got_time(time_started, time_limit):
            raise Exception

        if env.done():
            if env.get_robot(original_agent_id).credit > env.get_robot((original_agent_id + 1) % 2).credit:
                return float('inf')
            elif env.get_robot(original_agent_id).credit < env.get_robot((original_agent_id + 1) % 2).credit:
                return self.heuristic(env, original_agent_id)
            else:
                # TODO: in case of draw, return either inf or -inf when clarifications on the matter are given
                # for now, putting -inf
                return self.heuristic(env, original_agent_id)
        if depth == 0:
            return self.heuristic(env, original_agent_id)

        if agent_turn == original_agent_id:
            return self.max_value(env, agent_turn, time_limit, original_agent_id, time_started, depth - 1)
        else:
            return self.min_value(env, agent_turn, time_limit, original_agent_id, time_started, depth - 1)

    def max_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started, depth):
        successors = self.successors(env, agent_id)
        # print(f"function: max_value. turn: {agent_id} successors: ", successors)
        max_heuristic = float('-inf')

        for op, child in zip(successors[0], successors[1]):
            curr_heuristic = self.minimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started, depth)
            max_heuristic = max(max_heuristic, curr_heuristic)
        return max_heuristic

    def min_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started, depth):
        successors = self.successors(env, agent_id)
        # print(f"function: min_value. turn: {agent_id} successors: ", successors)
        min_heuristic = float('inf')

        for op, child in zip(successors[0], successors[1]):
            curr_heuristic = self.minimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started, depth)
            min_heuristic = min(min_heuristic, curr_heuristic)
        return min_heuristic


class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        raise NotImplementedError()


class AgentExpectimax(Agent):
    # TODO: section d : 1

    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    @staticmethod
    def got_time(start, time_limit):
        return check_time_limit(start, time_limit)

    def calculate_probabilities(self, successors):
        denominator = 0
        probabilities = {}
        global SPECIAL_OPS
        for op, _ in zip(successors[0], successors[1]):
            if op in SPECIAL_OPS:
                denominator += 2
            else:
                denominator += 1
        for op, _ in zip(successors[0], successors[1]):
            if op not in SPECIAL_OPS:
                probabilities[op] = 1 / denominator
            else:
                probabilities[op] = 2 / denominator
        return probabilities

    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start = time.time()
        depth = 1
        operation = ''
        # helps for printing the children
        # print("\n\n")

        # doing one step of expectimax since I want to return the operator
        # the expectimax function will return the max value of each child
        try:
            while True:
                # operators = env.get_legal_operators(agent_id)
                ops_children = self.successors(env, agent_id)
                children_values = []
                max_heuristic = float('-inf')

                for op, child in zip(ops_children[0], ops_children[1]):
                    curr_value = self.expectimax(child, (agent_id + 1) % 2, time_limit, agent_id, start, depth)
                    children_values.append((curr_value, op))
                    max_heuristic = max(max_heuristic, curr_value)

                # print the children heuristics and the operator
                # operators_and_values = [(j, operators[i]) for i, j in enumerate(children_values)]
                # print("the operators and their values for expectimax agent:")
                # print(children_values)

                max_values_operations = [tup[1] for tup in children_values if tup[0] == max_heuristic]
                operation = random.choice(max_values_operations)
                if max_heuristic == float('inf'):
                    raise Exception
                depth += 1

        except:
            # helps for printing the children
            # print("\n\n")
            return operation

    def expectimax(self, env: WarehouseEnv, agent_turn, time_limit, original_agent_id, time_started, depth):
        if not self.got_time(time_started, time_limit):
            raise Exception

        if env.done() or depth == 0:
            return self.heuristic(env, original_agent_id)

        if agent_turn == original_agent_id:
            return self.max_value(env, agent_turn, time_limit, original_agent_id, time_started, depth - 1)
        else:
            return self.expected_value(env, agent_turn, time_limit, original_agent_id, time_started, depth - 1)

    def max_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started, depth):
        successors = self.successors(env, agent_id)
        max_heuristic = float('-inf')

        for op, child in zip(successors[0], successors[1]):
            curr_heuristic = self.expectimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started,
                                             depth)
            max_heuristic = max(max_heuristic, curr_heuristic)
        return max_heuristic

    def expected_value(self, env: WarehouseEnv, agent_id, time_limit, original_agent_id, time_started, depth):
        successors = self.successors(env, agent_id)
        probabilities = self.calculate_probabilities(successors)
        expected_heuristic = 0

        for op, child in zip(successors[0], successors[1]):
            curr_heuristic = self.expectimax(child, (agent_id + 1) % 2, time_limit, original_agent_id, time_started,
                                             depth)

            expected_heuristic += curr_heuristic * probabilities[op]
        return expected_heuristic


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
