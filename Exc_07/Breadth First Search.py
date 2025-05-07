from collections import deque


def water_jug_bfs(jug1_capacity, jug2_capacity, target1, target2):
    # Define possible actions
    actions = [
        lambda x, y: (jug1_capacity, y),  # Fill jug1
        lambda x, y: (x, jug2_capacity),  # Fill jug2
        lambda x, y: (0, y),  # Empty jug1
        lambda x, y: (x, 0),  # Empty jug2
        lambda x, y: (x - min(x, jug2_capacity - y), y + min(x, jug2_capacity - y)),  # Pour jug1 to jug2
        lambda x, y: (x + min(y, jug1_capacity - x), y - min(y, jug1_capacity - x)),  # Pour jug2 to jug1
    ]

    # Initialize BFS structures
    queue = deque([(0, 0)])  # Start with both jugs empty
    visited = set([(0, 0)])  # Set to track visited states
    path = {(0, 0): None}  # To reconstruct the path

    while queue:
        jug1, jug2 = queue.popleft()

        # If we reach the target state, reconstruct the solution path
        if (jug1, jug2) == (target1, target2):
            result = []
            state = (jug1, jug2)
            while state is not None:
                result.append(state)
                state = path[state]
            result.reverse()  # Reverse to get the correct order of steps
            return result  # Return the solution path

        # Try all possible actions
        for action in actions:
            next_state = action(jug1, jug2)
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                path[next_state] = (jug1, jug2)  # Record the previous state for path reconstruction

    return None  # No solution found


# Example usage
jug1_capacity = 4
jug2_capacity = 3
target1 = 2
target2 = 0

solution = water_jug_bfs(jug1_capacity, jug2_capacity, target1, target2)

if solution:
    print("Solution Path:")
    for step in solution:
        print(step)
    print("Total Steps:", len(solution) - 1)  # Subtract 1 for the initial state
else:
    print("No solution found")
