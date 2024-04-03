from copy import deepcopy
import numpy as np
import mdp as m
actions_dict = {
    'UP': 0,
    'DOWN': 1,
    'RIGHT': 2,
    'LEFT': 3
}
def find_best_utility_for_state(mdp: m.MDP, U_bar, r, c):
    global actions_dict
    max_val = -np.inf
    # this loop is to check the maximum action
    for action in mdp.actions:
        val = 0
        # this loop is to sum up the probabilities that we do another action instead
        for prob_action in mdp.actions:
            next_state = mdp.step((r, c), prob_action)
            val += mdp.transition_function[action][actions_dict[prob_action]] * U_bar[next_state[0]][next_state[1]]
        if val > max_val:
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
                if r == 0 and c == 2:
                    print("debug")
                U_bar[r][c] = find_best_utility_for_state(mdp, U_bar, r, c)
                if np.abs(U_bar[r][c] - U[r][c]) > delta:
                    delta = np.abs(U_bar[r][c] - U[r][c])
        print(delta)
    return U
    # ========================


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    # TODO: Maybe I acciedentally let copilot implement this function, check if it's correct
    # U_bar = np.array(U)
    # policy = np.zeros((mdp.num_row, mdp.num_col), dtype=object)
    # for r in range(mdp.num_row):
    #     for c in range(mdp.num_col):
    #         if (r, c) in mdp.terminal_states:
    #             policy[r][c] = None
    #         elif mdp.board[r][c] == 'WALL':
    #             policy[r][c] = None
    #         else:
    #             max_val = -np.inf
    #             for action in mdp.actions:
    #                 next_state = mdp.step((r, c), action)
    #                 val = mdp.reward(next_state) + mdp.gamma * U_bar[next_state[0], next_state[1]]
    #                 if val > max_val:
    #                     max_val = val
    #                     policy[r][c] = action
    # return policy
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
