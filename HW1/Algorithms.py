import queue
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
        self.OPEN_QUEUE = queue.PriorityQueue()
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
                 f: float = None) -> None:
        self.state = state
        self.parent = parent
        self.g = g
        self.action = action
        self.depth = depth
        self.terminated = terminated
        self.f = f

    # TODO: check if these functions are useful
    def __lt__(self, other: 'Node') -> bool:
        if self.f == other.f:
            return self.state < other.state
        return self.f < other.f

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
        self.env = env
        self.OPEN_QUEUE = queue.PriorityQueue(maxsize =  env.ncol * env.nrow * 2 * 2) 

        self.env.reset()
        hmap = self.h_msap(self.env.get_initial_state())
        print(hmap)
        expanded_nodes = 0
        node = Node(state = self.env.get_initial_state(), parent = None, g = 0, action = 0, depth = 0, 
                    terminated = False, f = h_weight * self.h_msap(self.env.get_initial_state()))
        if self.env.is_final_state(node.state):
            return [], 0, 0
        self.OPEN_QUEUE.put(((node.f, node.state[0]), node))

        while self.OPEN_QUEUE:
            
            node = self.OPEN_QUEUE.get()[1]
            self.CLOSE.append(node)
            expanded_nodes += 1
            self.env.set_state_2(node.state)
            if node.terminated is True and not self.env.is_final_state(node.state):
                continue

            for action, (s, c, term) in self.env.succ(node.state).items():

                node_new_state, cost, terminated = self.env.step(action)
                f_val = (node.g + cost) * (1 - h_weight) +  self.h_msap(node_new_state) * h_weight
                new_node = Node(node_new_state, node, node.g + cost, action, 0, bool(terminated), f_val)
                
                if self.env.is_final_state(node_new_state):
                        return self.solution(new_node), new_node.g, expanded_nodes #check
                    
                # iterable_queue = self.queue_to_list(self.OPEN_QUEUE)
                open_iterable_queue = None
                if not self.OPEN_QUEUE.empty():
                    open_iterable_queue = list(self.OPEN_QUEUE.queue)
                    
                
                if (node_new_state not in [c.state for c in self.CLOSE]) and ((open_iterable_queue is None) or (node_new_state not in [n[1].state for n in open_iterable_queue])):
                    self.OPEN_QUEUE.put(((new_node.f, new_node.state[0]), new_node))
                elif node_new_state in [n[1].state for n in open_iterable_queue]:
                    for n in open_iterable_queue:
                        if n[1].state == node_new_state and n[1].f > new_node.f:
                            self.OPEN_QUEUE.remove(n)
                            self.OPEN_QUEUE.put(((new_node.f, new_node.state[0]), new_node))
                            break
                elif new_node in self.CLOSE:
                    for n in self.CLOSE:
                        if n.state == node_new_state and n.f > new_node.f:
                            self.CLOSE.remove(n)
                            self.OPEN_QUEUE.put(((new_node.f, new_node.state[0]), new_node))
                            break
                self.env.set_state_2(node.state)
                
        return [], 0, expanded_nodes
    
    def queue_to_list(self, priority_queue):
        priority_queue_list = []
        while not priority_queue.empty():
            item = priority_queue.get()[1]
            priority_queue_list.append(item)
    
    def h_msap(self, state: Tuple) -> int:
        min_dist = 1000000
        for goal in self.env.get_goal_states():
            row_diff = goal[0]//self.env.ncol - state[0]//self.env.ncol
            col_diff = goal[0]%self.env.ncol - state[0]%self.env.ncol
            dist = row_diff + col_diff
            if dist < min_dist:
                min_dist = dist
        
        d1_dist = self.env.d1[0]//self.env.ncol - state[0]//self.env.ncol + self.env.d1[0]%self.env.ncol - state[0]%self.env.ncol
                
        if d1_dist < min_dist:
            min_dist = d1_dist
        
        d2_dist = self.env.d2[0]//self.env.ncol - state[0]//self.env.ncol + self.env.d2[0]%self.env.ncol - state[0]%self.env.ncol
        if d2_dist < min_dist:
            min_dist = d2_dist
            
        return min_dist


class AStarEpsilonAgent():
    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: DragonBallEnv, epsilon: int) -> Tuple[List[int], float, int]:
        raise NotImplementedError
