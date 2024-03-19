import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict

from queue import Queue


class Node:
    def __init__(self, state, action_list, total_cost, terminated):
        self.state = state
        self.action_list = action_list
        self.total_cost = total_cost
        self.terminated = terminated

    def __lt__(self, other):
        return self.total_cost < other.total_cost

    def __eq__(self, other):
        return self.state == other.state

    def __hash__(self):
        return hash(self.state)


def manhattan_distance(env, state1, state2):
    row1, col1 = env.to_row_col(state1)
    row2, col2 = env.to_row_col(state2)
    return abs(row1 - row2) + abs(col1 - col2)


def h_msap(env, state):
    final_states = env.get_goal_states()

    d1, d2 = env.d1, env.d2
    active_balls = []
    if not state[1]:
        active_balls.append(d1)
    if not state[2]:
        active_balls.append(d2)

    locations = final_states + active_balls

    return min([manhattan_distance(env, state, loc) for loc in locations])


def h_msap_new(env, state):
    final_states = env.get_goal_states()

    d1, d2 = env.d1, env.d2
    active_balls = []
    if not state[1]:
        active_balls.append(d1)
    if not state[2]:
        active_balls.append(d2)

    if len(active_balls) == 0:
        return min([manhattan_distance(env, state, loc) for loc in final_states])
    else:
        return min([manhattan_distance(env, state, loc) for loc in active_balls])


class BFSAgent:
    def __init__(self) -> None:
        self.visited = set()
        self.queue = Queue()
        self.expanded = 0

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        self.visited.clear()
        self.queue.queue.clear()
        self.expanded = 0
        env.reset()

        first_state = env.get_initial_state()
        node = Node(first_state, [], 0, False)
        self.queue.put(node)

        while not self.queue.empty():
            node = self.queue.get()
            self.expanded += 1
            self.visited.add(node.state)

            env.set_state_2(node.state)

            if node.terminated and not env.is_final_state(node.state):
                continue

            for action in env.succ(node.state):
                new_state, cost, terminated = env.step(action)

                new_action_list = node.action_list + [action]
                new_total_cost = node.total_cost + cost
                new_node = Node(new_state, new_action_list, new_total_cost, terminated)

                if new_state not in self.visited and new_state not in [x.state for x in list(self.queue.queue)]:
                    if env.is_final_state(new_state):
                        return new_action_list, new_total_cost, self.expanded
                    self.queue.put(new_node)
                env.set_state_2(node.state)

        return [], 0, 0


class WeightedAStarAgent:
    def __init__(self) -> None:
        self.visited = set()
        self.queue = heapdict.heapdict()
        self.expanded = 0

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        self.visited.clear()
        self.queue.clear()
        self.expanded = 0
        env.reset()

        first_state = env.get_initial_state()
        node = Node(first_state, [], 0, False)
        self.queue[node] = ((1 - h_weight) * 0 + h_weight * h_msap(env, first_state), 0)

        while self.queue:
            node = self.queue.popitem()[0]
            self.visited.add(node)

            env.set_state_2(node.state)

            if env.is_final_state(node.state):
                return node.action_list, node.total_cost, self.expanded

            self.expanded += 1
            if node.terminated:
                continue

            for action in env.succ(node.state):
                env.set_state_2(node.state)
                new_state, cost, terminated = env.step(action)

                new_action_list = node.action_list + [action]
                new_total_cost = node.total_cost + cost
                new_node = Node(new_state, new_action_list, new_total_cost, terminated)

                closed = [x.state for x in list(self.visited)]
                if not (new_state in closed or new_state in [x.state for x in list(self.queue.keys())]):
                    self.queue[new_node] = ((1 - h_weight) * new_total_cost + h_weight * h_msap(env, new_state), new_node.state[0])

                elif new_state in [x.state for x in list(self.queue.keys())]:
                    if new_total_cost < [x[0] for x in list(self.queue.items()) if x[0].state == new_state][0].total_cost:
                        del self.queue[[x[0] for x in list(self.queue.items()) if x[0].state == new_state][0]]
                        self.queue[new_node] = ((1 - h_weight) * new_total_cost + h_weight * h_msap(env, new_state), new_node.state[0])

                else:
                    if new_total_cost < [x.total_cost for x in list(self.visited) if x.state == new_state][0]:
                        self.visited.remove([x for x in list(self.visited) if x.state == new_state][0])
                        self.queue[new_node] = ((1 - h_weight) * new_total_cost + h_weight * h_msap(env, new_state), new_node.state[0])

        return [], 0, 0


class AStarEpsilonAgent:
    def __init__(self) -> None:
        self.visited = set()
        self.queue = heapdict.heapdict()
        self.expanded = 0

    @staticmethod
    def choose_node(env, open_set, min_f_val, eps):
        copy_open = heapdict.heapdict(open_set)
        curr_f = min_f_val
        focal = heapdict.heapdict()
        while copy_open:
            curr_node, (curr_f, _) = copy_open.popitem()
            if curr_f > (1 + eps) * min_f_val:
                break
            focal[curr_node] = (h_msap_new(env, curr_node.state), curr_node.state[0])

        return focal.popitem()[0]

    def search(self, env: DragonBallEnv, epsilon) -> Tuple[List[int], float, int]:
        self.visited.clear()
        self.queue.clear()
        self.expanded = 0
        env.reset()

        first_state = env.get_initial_state()
        node = Node(first_state, [], 0, False)
        self.queue[node] = (0 + h_msap(env, first_state), 0)

        while self.queue:
            lowest_node = self.queue.peekitem()[0]
            node = self.choose_node(env, self.queue, lowest_node.total_cost + h_msap(env, lowest_node.state), epsilon)
            self.queue.pop(node)

            self.visited.add(node)

            env.set_state_2(node.state)

            if env.is_final_state(node.state):
                return node.action_list, node.total_cost, self.expanded

            self.expanded += 1
            if node.terminated:
                continue

            for action in env.succ(node.state):
                env.set_state_2(node.state)
                new_state, cost, terminated = env.step(action)

                new_action_list = node.action_list + [action]
                new_total_cost = node.total_cost + cost
                new_node = Node(new_state, new_action_list, new_total_cost, terminated)

                closed = [x.state for x in list(self.visited)]
                if not (new_state in closed or new_state in [x.state for x in list(self.queue.keys())]):
                    self.queue[new_node] = (new_total_cost + h_msap(env, new_state), new_node.state[0])

                elif new_state in [x.state for x in list(self.queue.keys())]:
                    if new_total_cost < [x[0] for x in list(self.queue.items()) if x[0].state == new_state][0].total_cost:
                        del self.queue[[x[0] for x in list(self.queue.items()) if x[0].state == new_state][0]]
                        self.queue[new_node] = (new_total_cost + h_msap(env, new_state), new_node.state[0])

                else:
                    if new_total_cost < [x.total_cost for x in list(self.visited) if x.state == new_state][0]:
                        self.visited.remove([x for x in list(self.visited) if x.state == new_state][0])
                        self.queue[new_node] = (new_total_cost + h_msap(env, new_state), new_node.state[0])

        return [], 0, 0
