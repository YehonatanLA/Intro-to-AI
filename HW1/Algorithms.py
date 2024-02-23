import numpy as np

from DragonBallEnv import DragonBallEnv
from typing import List, Tuple
import heapdict


class Agent:
    def __init__(self) -> None:
        self.env = None
        self.OPEN = []
        self.CLOSE = []
        self.path = []


class Node:
    def __init__(self, state: Tuple, parent: 'Node', action: int, cost: float, depth: int, h: float = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.depth = depth
        self.h = h

    # TODO: check if these functions are useful
    def __lt__(self, other: 'Node') -> bool:
        return self.cost + self.h < other.cost + other.h

    def __eq__(self, other: 'Node') -> bool:
        return self.state == other.state

    def __str__(self) -> str:
        return f"Node: state={self.state}, action={self.action}, cost={self.cost}, depth={self.depth}, h={self.h}"


class BFSAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def search(self, env: DragonBallEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError


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
