from copy import deepcopy
import numpy as np
import itertools
from tqdm import tqdm


def float_equal(a, b):
    return abs(b - a) < 1e-6


# need to change!
def transition_prob(mdp, s_tag, a, s):
    if s in mdp.terminal_states:
        return 0
    map_ = {'UP': 0, 'DOWN': 1, 'RIGHT': 2, 'LEFT': 3}

    actions = [b for b in mdp.actions if mdp.step(s, b) == s_tag]
    return sum([mdp.transition_function[a][map_[b]] for b in actions])


def p_val_for_action(mdp, s, a, U):
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    return sum([transition_prob(mdp, s_tag, a, s) * U[s_tag[0]][s_tag[1]] for s_tag in states])


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    U_t = deepcopy(U_init)

    while True:
        # local vars
        U = deepcopy(U_t)
        delta = 0

        # get all states
        states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
        states = [s for s in states if mdp.board[s[0]][s[1]] != 'WALL']

        # for each state
        for (r, c) in states:

            max_val = max([p_val_for_action(mdp, (r, c), action, U) for action in mdp.actions])
            U_t[r][c] = max_val * mdp.gamma + float(mdp.board[r][c])

            delta = max(delta, abs(U_t[r][c] - U[r][c]))
            if float(mdp.board[0][0]) == -0.37:
                print(delta)
            if float_equal(1, mdp.gamma) and float_equal(delta, 0) or (
                    not float_equal(1, mdp.gamma)) and delta < epsilon * (1 - mdp.gamma) / mdp.gamma:
                return U_t


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    policy = deepcopy(U)
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    for (r, c) in states:
        # in case of wall
        if mdp.board[r][c] == 'WALL':
            policy[r][c] = None
            continue

        # find the best action arg
        vals = [(p_val_for_action(mdp, (r, c), action, U), action) for action in mdp.actions]
        _, best_action = max(vals, key=lambda x: x[0])
        policy[r][c] = best_action

    return policy


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    states = [s for s in states if mdp.board[s[0]][s[1]] != 'WALL']

    p_matrix = []
    for s in states:
        p_matrix.append([transition_prob(mdp, s_tag, policy[s[0]][s[1]], s) for s_tag in states])

    p_matrix = np.array(p_matrix)
    r_vector = np.array([float(mdp.board[s[0]][s[1]]) for s in states])

    u_vector = np.linalg.inv(np.eye(len(states)) - mdp.gamma * p_matrix) @ r_vector.T

    u_to_return = []
    idx = 0
    for i in range(mdp.num_row):
        row = []
        for j in range(mdp.num_col):
            if mdp.board[i][j] == 'WALL':
                row.append(0)
            else:
                row.append(u_vector[idx])
                idx += 1
        u_to_return.append(row)

    return u_to_return


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    states = [s for s in states if mdp.board[s[0]][s[1]] != 'WALL']

    while True:
        U = policy_evaluation(mdp, policy_init)
        unchanged = True
        for (r, c) in states:

            val_action_lst = [(p_val_for_action(mdp, (r, c), action, U), action) for action in mdp.actions]

            max_val, best_action = max(val_action_lst, key=lambda x: x[0])
            policy_val = p_val_for_action(mdp, (r, c), policy_init[r][c], U)

            if max_val > policy_val:
                policy_init[r][c] = best_action
                unchanged = False

        if unchanged:
            return policy_init


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U, epsilon=1e-3, to_print=True):  # You can add more input parameters as needed

    def is_equal(a, b, p=4):
        return abs(round(a, p) - round(b, p)) <= epsilon

    translate = {'UP': "↑", 'DOWN': "↓", 'RIGHT': "→", 'LEFT': "←"}
    policies = [["" for _ in range(mdp.num_col)] for _ in range(mdp.num_row)]
    precision = int(np.log10(1 / epsilon) + 1)

    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    states = [s for s in states if mdp.board[s[0]][s[1]] != 'WALL' and s not in mdp.terminal_states]

    for (r, c) in states:
        p_vals_lst = [(p_val_for_action(mdp, (r, c), action, U), action) for action in mdp.actions]
        max_val, _ = max(p_vals_lst, key=lambda x: x[0])

        new_val_lst = [(val, action) for val, action in p_vals_lst if is_equal(val, max_val, precision)]
        policies[r][c] = "".join([translate[action] for _, action in new_val_lst])

    if to_print:
        mdp.print_policy(policies)
    mull = 1
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if len(policies[i][j]) > 0:
                mull *= len(policies[i][j])
    return mull, policies


def get_policy_for_different_rewards(mdp, epsilon=1e-3):
    # ====== YOUR CODE: ======
    initial_u = np.zeros((mdp.num_row, mdp.num_col)).tolist()

    r_lst = np.arange(-0.40, 5.01, 0.01)
    states = list(itertools.product(range(mdp.num_row), range(mdp.num_col)))
    states = [s for s in states if mdp.board[s[0]][s[1]] != 'WALL' and s not in mdp.terminal_states]
    policies_lst = []
    # for R in tqdm(r_lst):
    for R in r_lst:
        curr_mdp = deepcopy(mdp)
        for (r, c) in states:
            curr_mdp.board[r][c] = str(round(R,2))  # need to change
        curr_u = value_iteration(curr_mdp, initial_u)
        _, policies = get_all_policies(curr_mdp, curr_u, to_print=False)
        policies_lst.append(policies)

    policy_ranges = []
    start = 0
    for i in tqdm(range(1, len(r_lst))):
        if policies_lst[i] != policies_lst[start]:
            policy_ranges.append((r_lst[start], r_lst[i - 1], policies_lst[start]))
            start = i
    policy_ranges.append((r_lst[start], r_lst[-1], policies_lst[start]))


    for r_start, r_end, policy in policy_ranges:
        if r_start == -5.0:
            print(f"R(s) < {round(r_end, 2)}:\n---------------------------------")
        elif r_end == 5.0:
            print(f"{round(r_start, 2)} < R(s):\n---------------------------------")
        else:
            print(f"{round(r_start, 2)} < R(s) < {round(r_end, 2)}:\n---------------------------------")
        mdp.print_policy(policy)
