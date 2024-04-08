import itertools
import math
from copy import deepcopy
import numpy as np
import mdp as m

actions_dict = {
    'UP': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'LEFT': 3
}
ROWS = 3
COLUMNS = 4


def iterate_over_action(mdp: m.MDP, U_bar, r, c, action):
    val = 0
    # this loop is to sum up the probabilities that we do another action instead
    for prob_action in mdp.actions:
        next_state = mdp.step((r, c), prob_action)
        t = mdp.transition_function[action][actions_dict[prob_action]]
        b = U_bar[next_state[0]][next_state[1]]
        val += mdp.transition_function[action][actions_dict[prob_action]] * U_bar[next_state[0]][next_state[1]]
    return val


def find_best_action_for_state(mdp: m.MDP, U, r, c):
    global actions_dict
    max_val = -np.inf
    max_action = None

    for action in mdp.actions:
        val = iterate_over_action(mdp, U, r, c, action)
        if round(val, 2) > round(max_val, 2):
            max_action = action
            max_val = val
    return max_action


def find_best_actions_for_state(mdp: m.MDP, U, r, c, epsilon=10 ** (-3)):
    global actions_dict
    max_val = -np.inf
    actions_list = []
    for action in mdp.actions:
        val = iterate_over_action(mdp, U, r, c, action)
        if val >= max_val - epsilon:
            if val > max_val + epsilon:
                actions_list.clear()
            action_arrow = '\u2192'
            if action == 'UP':
                action_arrow = '\u2191'
            elif action == 'DOWN':
                action_arrow = '\u2193'
            elif action == 'LEFT':
                action_arrow = '\u2190'
            actions_list.append(action_arrow)
            max_val = val

    return actions_list, len(actions_list)


def find_best_utility_for_state(mdp: m.MDP, U_bar, r, c):
    global actions_dict
    max_val = -np.inf
    # this loop is to check the maximum action
    for action in mdp.actions:
        val = iterate_over_action(mdp, U_bar, r, c, action)
        if round(val, 2) > round(max_val, 2):
            max_val = val
    return float(float(mdp.board[r][c]) + mdp.gamma * max_val)
    # return float(U_bar[r][c] + mdp.gamma * max_val)


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U = None
    U_bar = deepcopy(U_init)
    for terminal_state in mdp.terminal_states:
        U_bar[terminal_state[0]][terminal_state[1]] = float(mdp.board[terminal_state[0]][terminal_state[1]])
    delta = np.inf
    while delta > epsilon * (1 - mdp.gamma) / mdp.gamma:
        U = deepcopy(U_bar)
        delta = 0
        # for each state s in S:
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                # if s is terminal state or wall then no utility update
                if (r, c) in mdp.terminal_states:
                    continue
                if mdp.board[r][c] == 'WALL':
                    continue

                U_bar[r][c] = find_best_utility_for_state(mdp, U_bar, r, c)

        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if round(np.abs(U_bar[r][c] - U[r][c]), 5) > delta:
                    delta = np.abs(U_bar[r][c] - U[r][c])
        if float(mdp.board[0][0]) == -0.37:
            mdp.print_utility(U_bar)

    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    global ROWS, COLUMNS
    policy = [[""] * COLUMNS for _ in range(ROWS)]

    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if (r, c) in mdp.terminal_states:
                continue
            if mdp.board[r][c] == 'WALL':
                continue

            policy[r][c] = find_best_action_for_state(mdp, U, r, c)
    return policy
    # ========================


def policy_evaluation(mdp, policy):
    I = np.eye(mdp.num_row * mdp.num_col)
    policy_mat = np.zeros((mdp.num_row * mdp.num_col, mdp.num_row * mdp.num_col))
    reward = np.zeros(mdp.num_row * mdp.num_col)
    map_actions = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if mdp.board[r][c] == 'WALL':
                continue
            state = r * mdp.num_col + c
            reward[state] = float(mdp.board[r][c])
            if (r, c) in mdp.terminal_states:
                continue

            for action in mdp.actions:
                next_state = mdp.step((r, c), action)
                next_state_pos = next_state[0] * mdp.num_col + next_state[1]
                wanted_action = map_actions[policy[r][c]]
                cur_action = map_actions[action]
                policy_mat[state][next_state_pos] += mdp.transition_function[policy[r][c]][cur_action]

    my_mat = np.add(I, np.dot(-mdp.gamma, policy_mat))
    res_vector = np.linalg.solve(my_mat, reward)
    return res_vector.reshape((mdp.num_row, mdp.num_col))
    # utility = np.empty((mdp.num_row, mdp.num_col))


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    unchanged = False
    while not unchanged:
        U = policy_evaluation(mdp, policy_init)
        unchanged = True
        # for each state s in S:
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if (r, c) in mdp.terminal_states or mdp.board[r][c] == "WALL":
                    continue
                max_action_value = iterate_over_action(mdp, U, r, c, find_best_action_for_state(mdp, U, r, c))
                predicted_action_value = iterate_over_action(mdp, U, r, c, policy_init[r][c])
                if round(max_action_value, 2) > round(predicted_action_value, 2):
                    policy_init[r][c] = find_best_action_for_state(mdp, U, r, c)
                    unchanged = False
    return policy_init
    # ========================


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    global ROWS, COLUMNS
    policy = [[""] * COLUMNS for _ in range(ROWS)]
    policies_num = 1
    for r in range(mdp.num_row):
        for c in range(mdp.num_col):
            if (r, c) in mdp.terminal_states:
                continue
            if mdp.board[r][c] == 'WALL':
                continue
            actions_cell, num_actions_cell = find_best_actions_for_state(mdp, U, r, c, epsilon)
            actions_str = ''.join(actions_cell)
            policy[r][c] = actions_str
            policies_num *= num_actions_cell

    # mdp.print_policy(policy)
    return policies_num, policy
    # ========================


def get_policy_for_different_rewards(mdp, epsilon=10 ** (-3)):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    MINIMAL_REWARD = -5
    MAXIMAL_REWARD = 5
    policies_lists = []
    rewards = []
    curr_reward = MINIMAL_REWARD
    while curr_reward <= MAXIMAL_REWARD:
        initial_u = np.zeros((mdp.num_row, mdp.num_col)).tolist()
        for r in range(mdp.num_row):
            for c in range(mdp.num_col):
                if (r, c) in mdp.terminal_states:
                    continue
                if mdp.board[r][c] == 'WALL':
                    continue
                mdp.board[r][c] = str(curr_reward)
        utility = value_iteration(mdp, initial_u)
        _, curr_policies = get_all_policies(mdp, utility, epsilon)

        if len(policies_lists) == 0 or curr_policies != policies_lists[-1]:
            policies_lists.append(curr_policies)
            rewards.append(curr_reward)
            print("Reward: ", curr_reward)
            mdp.print_policy(curr_policies)
        curr_reward += 0.01
        curr_reward = round(curr_reward, 2)
        print("Current reward: ", curr_reward)
        if curr_reward == -0.37:
            print("HERE")
    return rewards
    # ========================
