from collections import deque

def water_jug_bfs(jug1_capacity, jug2_capacity, target1, target2):
    actions = [
        lambda x, y: (jug1_capacity, y),
        lambda x, y: (x, jug2_capacity),
        lambda x, y: (0, y),
        lambda x, y: (x, 0),
        lambda x, y: (x - min(x, jug2_capacity - y), y + min(x, jug2_capacity - y)),
        lambda x, y: (x + min(y, jug1_capacity - x), y - min(y, jug1_capacity - x)),
    ]

    queue = deque([(0, 0)])
    visited = set(queue)
    path = { (0, 0): None }

    while queue:
        jug1, jug2 = queue.popleft()

        if (jug1, jug2) == (target1, target2):
            # Fix: Correct path reconstruction
            result = []
            state = (jug1, jug2)
            while state is not None:
                result.append(state)
                state = path[state]
            return list(reversed(result))

        for action in actions:
            next_state = action(jug1, jug2)
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
                path[next_state] = (jug1, jug2)

    return None # No solution found

# Example usage
jug1_capacity = 4
jug2_capacity = 3
target1 = 4
target2 = 1

solution = water_jug_bfs(jug1_capacity, jug2_capacity, target1, target2)
if solution:
    print("Solution Path:", solution)
    print("Solution Path:", len(solution))
else:
    print("No solution found")
