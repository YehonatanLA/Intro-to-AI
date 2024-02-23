import time

import numpy as np
from IPython.core.display import clear_output

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class Agent:
    def __init__(self) -> None:
        self.env = None
        self.OPEN = []
        self.CLOSE = []
        self.path = []

    def solution(self, node: 'Node') -> List[int]:
        path = []
        while node.parent:
            path.append(node.action)
            node = node.parent
        return path[::-1]


class Node:
    def __init__(self, state: Tuple, parent: 'Node', g: int, action: int, depth: int, terminated: bool,
                 h: float = None) -> None:
        self.state = state
        self.parent = parent
        self.g = g
        self.action = action
        self.depth = depth
        self.terminated = terminated
        self.h = h

    # TODO: check if these functions are useful
    def __lt__(self, other: 'Node') -> bool:
        return self.g + self.h < other.g + other.h

    def __eq__(self, other: 'Node') -> bool:
        return self.state == other.state

    def __str__(self) -> str:
        return f"Node: state={self.state}, cost={self.g}, depth={self.depth}, h={self.h}"


def print_solution(actions, env: DragonBallEnv) -> None:
    env.reset()
    total_cost = 0
    print(env.render())
    print(f"Timestep: {1}")
    print(f"State: {env.get_state()}")
    print(f"Action: {None}")
    print(f"Cost: {0}")
    time.sleep(1)

    for i, action in enumerate(actions):
        state, cost, terminated = env.step(action)
        total_cost += cost
        clear_output(wait=True)

        print(env.render())
        print(f"Timestep: {i + 2}")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Cost: {cost}")
        print(f"Total cost: {total_cost}")

        time.sleep(1)

        if terminated is True:
            break


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: DragonBallEnv) -> Tuple[List[int], int, int]:
        self.env = env

        self.env.reset()

        expanded_nodes = 0
        node = Node(self.env.get_initial_state(), None, 0, 0, 0, False)
        if self.env.is_final_state(node.state):
            return [], 0, 0
        self.OPEN = [node]

        while self.OPEN:
            node = self.OPEN.pop(0)
            self.CLOSE.append(node.state)
            expanded_nodes += 1
            self.env.set_state_2(node.state)
            if node.terminated is True and not self.env.is_final_state(node.state):
                continue

            for action, (s, c, term) in self.env.succ(node.state).items():

                node_new_state, cost, terminated = self.env.step(action)
                new_node = Node(node_new_state, node, node.g + cost, action, 0, bool(terminated))

                if node_new_state not in self.CLOSE and node_new_state not in [n.state for n in self.OPEN]:
                    if self.env.is_final_state(node_new_state):
                        return self.solution(new_node), new_node.g, expanded_nodes
                    self.OPEN.append(new_node)
                self.env.set_state_2(node.state)

        return [], 0, expanded_nodes


class WeightedAStarAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: DragonBallEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
