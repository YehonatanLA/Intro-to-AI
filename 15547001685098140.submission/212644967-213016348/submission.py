from Agent import Agent, AgentGreedy
from WarehouseEnv import *
import random
import time
import numpy as np


# TODO: section a : 3
def dist_to_station(env: WarehouseEnv, robot: Robot):
    r = robot.position
    c0 = env.charge_stations[0].position
    c1 = env.charge_stations[1].position
    return min(manhattan_distance(r, c0), manhattan_distance(r, c1))


def dist_to_package(env: WarehouseEnv, robot: Robot):
    p0 = env.packages[0]
    p1 = env.packages[1]
    dist0 = manhattan_distance(robot.position, p0.position) if p0.on_board else np.inf
    dist1 = manhattan_distance(robot.position, p1.position) if p1.on_board else np.inf
    return min(dist0, dist1)


def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    total = 15 * robot.credit + 16 * robot.battery

    if robot.package is not None:  # if we can take a package:
        total += 15  # we want to take it

    if robot.package is not None:   # if we are carrying a package - first, we want to deliver it
        total += -manhattan_distance(robot.position, robot.package.destination)
    else:                           # if we are not carrying a package:
        if robot.battery <= 5:              # and we have low battery:
            total += -dist_to_station(env, robot)   # we want to charge
        else:                               # and we have high battery:
            total += 3 * (-dist_to_package(env, robot))   # we want to go to the closest package

    return total


class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def mini_max(self, env: WarehouseEnv, agent_id, d, time_limit, start_t, turn):
        if time.time()-start_t > 0.98*time_limit:
            raise TimeoutError
        if env.done() or d == 0:
            return self.heuristic(env, agent_id)
        if turn == agent_id:
            ops, children = self.successors(env, agent_id)
            CurMax = -np.inf
            for _, next_child in zip(ops, children):
                tmp = self.mini_max(next_child, agent_id, d-1, time_limit, start_t, 1-agent_id)
                CurMax = max(tmp, CurMax)
            return CurMax
        else:
            ops, children = self.successors(env, (agent_id+1) % 2)
            CurMin = np.inf
            for _, next_child in zip(ops, children):
                tmp = self.mini_max(next_child, agent_id, d-1, time_limit, start_t, agent_id)
                CurMin = min(CurMin, tmp)
            return CurMin
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_t, d, op = time.time(), 0, 'park'
        while True:
            try:
                result, max_res = [], []
                max_val = -np.inf
                ops, children = self.successors(env, agent_id)
                for next_op, next_child in zip(ops, children):
                    curr_val = self.mini_max(next_child, agent_id, d, time_limit, start_t, 1-agent_id)
                    result.append((curr_val, next_op))
                    max_val = max(curr_val, max_val)
                for curr_val, curr_op in result:
                    if curr_val == max_val:
                        max_res.append(curr_op)
                op = random.choice(max_res)
                d += 1
            except TimeoutError:
                return op
            
class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def alpha_beta(self, env: WarehouseEnv, agent_id, d, time_limit, start_t, turn, alpha, beta):
        if time.time()-start_t > 0.98*time_limit:
            raise TimeoutError
        if env.done() or d==0:
            return self.heuristic(env, agent_id)
        if turn == agent_id:
            ops, children = self.successors(env,agent_id)
            CurMax = -np.inf
            for _, next_child in zip(ops,children):
                tmp = self.alpha_beta(next_child, agent_id, d-1, time_limit, start_t, 1-agent_id, alpha, beta)
                CurMax = max(tmp, CurMax)
                alpha = max(CurMax, alpha)
                if CurMax >= beta:
                    return np.inf
            return CurMax
        else:
            ops, children = self.successors(env,(agent_id+1)%2)
            CurMin = np.inf
            for _,next_child in zip(ops, children):
                tmp = self.alpha_beta(next_child, agent_id, d-1, time_limit, start_t, 1-agent_id, alpha, beta)
                CurMin = min(CurMin, tmp)
                beta = min(CurMin, beta)
                if CurMin <= alpha:
                    return -np.inf
            return CurMin
        
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_t, d, op = time.time(), 0, 'park'
        while True:
            try:
                result, max_res = [], []
                max_val = -np.inf
                ops, children = self.successors(env, agent_id)
                for next_op, next_child in zip(ops, children):
                    curr_val = self.alpha_beta(next_child, agent_id, d, time_limit, start_t, 1-agent_id, -np.inf, np.inf)
                    result.append((curr_val, next_op))
                    max_val = max(curr_val, max_val)
                for curr_val, curr_op in result:
                    if curr_val == max_val:
                        max_res.append(curr_op)
                op = random.choice(max_res)
                d += 1
            except TimeoutError:
                return op


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)

    def prob(self, env: WarehouseEnv, enemy_id: int):
        is_prob = False
        operators, children = self.successors(env, enemy_id)

        c_x = []
        for c in children:
            if c.get_robot(enemy_id).position == c.charge_stations[0].position \
             or c.get_robot(enemy_id).position == c.charge_stations[1].position:
                c_x.append((c, 2))
                is_prob = True
            else:
                c_x.append((c, 1))

        total = sum([x for _, x in c_x])
        c_p = [(c, x/total) for c, x in c_x]
        return is_prob, c_p

    def RB_Expectimax(self, env: WarehouseEnv, agent_id, D, turn, stop_t):
        if time.time() > stop_t:
            raise TimeoutError
        if env.done() or D == 0:
            return self.heuristic(env, agent_id)
        if turn == agent_id:
            cur_max = -np.inf
            for c in self.successors(env, agent_id)[1]:
                v = self.RB_Expectimax(c, agent_id, D - 1, 1 - turn, stop_t)
                cur_max = max(v, cur_max)
            return cur_max
        else:   # turn != agent_id:
            is_prob, c_p = self.prob(env, (agent_id + 1) % 2)
            if is_prob:
                return sum([p * self.RB_Expectimax(c, agent_id, D - 1, 1 - turn, stop_t) for c, p in c_p])
            else:
                cur_min = np.inf
                for c, _ in c_p:
                    v = self.RB_Expectimax(c, agent_id, D - 1, 1 - turn, stop_t)
                    cur_min = min(v, cur_min)
                return cur_min


    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_t, D, chosen_operator = time.time(), 0, 'park'
        stop_t = start_t + 0.98 * time_limit
        while True:
            try:
                results = []
                max_value = -np.inf
                ops, children = self.successors(env, agent_id)
                for op, child in zip(ops, children):
                    value = self.RB_Expectimax(child, agent_id, D, 1 - agent_id, stop_t)
                    results.append((op, value))
                    max_value = max(value, max_value)

                chosen_operator = random.choice([op for op, value in results if value == max_value])
                D += 1
            except TimeoutError:
                return chosen_operator


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