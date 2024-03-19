import math

def calculate_cost(cell):
    cost_map = {'S': 1, 'G': 1, 'D': 1, 'F': 10, 'T': 3, 'A': 2, 'L': 1}
    return cost_map[cell]

def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_goal_positions(board):
    goal_positions = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'G' or board[i][j] == 'D':
                goal_positions.append((i, j))
    return goal_positions

def heuristic(state, goal_positions):
    min_distance = float('inf')
    for goal_pos in goal_positions:
        distance = manhattan_distance(state, goal_pos)
        min_distance = min(min_distance, distance)
    return min_distance

def print_board(board, current_position):
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if (i, j) == current_position:
                print('X', end=' ')
            else:
                print(cell, end=' ')
        print()

def ida_star_search(board):
    def search(state, g, f_limit, path, visited, passed_d):
        nonlocal expanded_nodes
        h_value = heuristic(state, goal_positions)
        f_value = g + h_value

        if f_value > f_limit:
            return f_value

        if state in goal_positions and board[state[0]][state[1]] == 'D':
            passed_d.add(state)

        if state in goal_positions:
            path.append(state)
            print_board(board, state)
            print("f_limit:", f_limit)
            print("Path:", path)
            if len(passed_d) == 2:
                return -1  # Goal state reached

        min_cost = float('inf')
        for move in moves:
            new_state = (state[0] + move[0], state[1] + move[1])
            if 0 <= new_state[0] < len(board) and 0 <= new_state[1] < len(board[0]) and new_state not in visited:
                if board[new_state[0]][new_state[1]] != 'W' and (move[0] == 0 or move[1] == 0):
                    expanded_nodes.append(new_state)
                    path.append(new_state)
                    visited.add(new_state)
                    cost = search(new_state, g + calculate_cost(board[new_state[0]][new_state[1]]), f_limit, path, visited, passed_d)
                    if cost == -1:
                        return -1  # Goal state reached
                    path.pop()
                    visited.remove(new_state)
                    min_cost = min(min_cost, cost)

        return min_cost

    goal_positions = get_goal_positions(board)
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    start_state = None
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'S':
                start_state = (i, j)

    f_limit = heuristic(start_state, goal_positions)
    path = [start_state]
    expanded_nodes = [start_state]
    visited = set([start_state])
    passed_d = set()

    print_board(board, start_state)
    print("f_limit:", f_limit)
    print("Path:", path)

    while True:
        cost = search(start_state, 0, f_limit, path, visited, passed_d)
        if cost == -1:
            return path  # Goal state reached

        if cost == float('inf'):
            return None  # No solution

        f_limit = cost

# Example usage
board = ["SFFF",
         "FDFF",
         "FFFD",
         "FFFG"]

result = ida_star_search(board)

if result:
    print("IDA* path:", result)
else:
    print("No solution found.")
