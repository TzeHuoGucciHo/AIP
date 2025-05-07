def water_jug_dfs(jug1_capacity, jug2_capacity, target1, target2):
    # Define possible actions
    actions = [
        lambda x, y: (x + min(y, jug1_capacity - x), y - min(y, jug1_capacity - x)),  # pour jug2 to jug1
        lambda x, y: (x - min(x, jug2_capacity - y), y + min(x, jug2_capacity - y)),  # pour jug1 to jug2
        lambda x, y: (x, 0),  # empty jug2
        lambda x, y: (0, y),  # empty jug1
        lambda x, y: (x, jug2_capacity),  # fill jug2
        lambda x, y: (jug1_capacity, y),  # fill jug1
    ]

    stack = [(0, 0)]
    visited = set([(0, 0)])
    path = {(0, 0): None}

    while stack:
        jug1, jug2 = stack.pop()

        if (jug1, jug2) == (target1, target2):
            # Reconstruct path from target to start
            result = []
            state = (jug1, jug2)
            while state is not None:
                result.append(state)
                state = path[state]
            result.reverse()
            return result

        for action in actions:
            next_state = action(jug1, jug2)
            if next_state not in visited:
                visited.add(next_state)
                stack.append(next_state)
                path[next_state] = (jug1, jug2)

    return None  # No solution found

# Example usage
jug1_capacity = 4
jug2_capacity = 3
target1 = 2
target2 = 0

solution = water_jug_dfs(jug1_capacity, jug2_capacity, target1, target2)

if solution:
    print("Solution Path (DFS):")
    for step in solution:
        print(step)
    print("Total Steps:", len(solution) - 1)
else:
    print("No solution found")
