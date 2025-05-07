import heapq
import matplotlib.pyplot as plt
import numpy as np
import time


def manhattan_distance(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    """Calculate the Euclidean distance between two points"""
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def bfs(maze, start, goal):
    """Breadth-First Search to find the shortest path from start to goal"""
    rows, cols = maze.shape
    queue = [start]
    came_from = {start: None}
    while queue:
        current = queue.pop(0)
        if current == goal:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0 and neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current
    return None


def dfs(maze, start, goal):
    """Depth-First Search to find the shortest path from start to goal"""
    rows, cols = maze.shape
    stack = [start]
    came_from = {start: None}
    while stack:
        current = stack.pop()
        if current == goal:
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0 and neighbor not in came_from:
                stack.append(neighbor)
                came_from[neighbor] = current
    return None


def find_path_using_a_star(maze, start, goal, distance_func=manhattan_distance, left_bias=False):
    """A* algorithm to find the shortest path from start to goal in the maze with optional left bias"""
    rows, cols = maze.shape
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: distance_func(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            # Reconstruct the path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        # Apply left bias by modifying neighbor order
        if left_bias:
            neighbors = [(current[0] - 1, current[1]), (current[0] + 1, current[1]), (current[0], current[1] - 1),
                         (current[0], current[1] + 1)]

        for neighbor in neighbors:
            r, c = neighbor
            if 0 <= r < rows and 0 <= c < cols and maze[r][c] == 0:
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + distance_func(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None


def visualize_maze_with_path(maze, path, start, goal, seed_value, num_obstacles, algorithm_name, distance_func_name):
    """Visualize the maze and the path found by the algorithm"""
    plt.figure(figsize=(12, 8))
    plt.imshow(maze, cmap='binary')
    if path:
        y, x = zip(*path)
        plt.plot(x, y, color='blue', linewidth=2, label='Path')
    plt.scatter(start[1], start[0], c='green', label='Start')
    plt.scatter(goal[1], goal[0], c='red', label='Goal')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.title(
        f"{algorithm_name} Pathfinding in Maze (Seed: {seed_value}, Obstacles: {num_obstacles}, {distance_func_name})")

    # Display the seed value and number of obstacles in the plot
    plt.text(0.5, -1, f"Seed: {seed_value}, Obstacles: {num_obstacles}", ha='center', va='center', fontsize=12,
             color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.show()


def generate_random_obstacles(maze, num_obstacles=100, seed=69):
    """Randomly place obstacles in the maze"""
    np.random.seed(seed)
    for _ in range(num_obstacles):
        r, c = np.random.randint(0, maze.shape[0]), np.random.randint(0, maze.shape[1])
        maze[r][c] = 1


def clear_surrounding_area(maze, start, goal, area_size=3):
    """Ensure that start and goal have a clear surrounding area"""
    for dx in range(-1, area_size - 1):
        for dy in range(-1, area_size - 1):
            sr, sc = start[0] + dx, start[1] + dy
            gr, gc = goal[0] + dx, goal[1] + dy
            if 0 <= sr < maze.shape[0] and 0 <= sc < maze.shape[1]:
                maze[sr][sc] = 0
            if 0 <= gr < maze.shape[0] and 0 <= gc < maze.shape[1]:
                maze[gr][gc] = 0


def get_random_free_position(maze):
    """Get a random free position in the maze (not an obstacle)"""
    rows, cols = maze.shape
    while True:
        r, c = np.random.randint(0, rows), np.random.randint(0, cols)
        if maze[r][c] == 0:
            return (r, c)


# --- MAIN SCRIPT ---

# Set maze dimensions
maze_height = 50
maze_width = 100
maze = np.zeros((maze_height, maze_width), dtype=int)

# Set the seed value
seed_value = 2

# Set the number of obstacles
num_obstacles = 2000

# Generate random obstacles in the maze using the seed
generate_random_obstacles(maze, num_obstacles=num_obstacles, seed=seed_value)

# Define random start and goal points
start = get_random_free_position(maze)
goal = get_random_free_position(maze)

# Clear space around start and goal points
clear_surrounding_area(maze, start, goal)

# --- Run BFS ---
start_time = time.time()
bfs_path = bfs(maze, start, goal)
bfs_time = time.time() - start_time
visualize_maze_with_path(maze, bfs_path, start, goal, seed_value, num_obstacles, "BFS", "N/A")
print(f"BFS Processing Time: {bfs_time:.4f} seconds")

# --- Run DFS ---
start_time = time.time()
dfs_path = dfs(maze, start, goal)
dfs_time = time.time() - start_time
visualize_maze_with_path(maze, dfs_path, start, goal, seed_value, num_obstacles, "DFS", "N/A")
print(f"DFS Processing Time: {dfs_time:.4f} seconds")

# --- Run A* with Manhattan Distance ---
start_time = time.time()
a_star_path_manhattan = find_path_using_a_star(maze, start, goal, distance_func=manhattan_distance)
a_star_time_manhattan = time.time() - start_time
visualize_maze_with_path(maze, a_star_path_manhattan, start, goal, seed_value, num_obstacles, "A*", "Manhattan")
print(f"A* with Manhattan Distance Processing Time: {a_star_time_manhattan:.4f} seconds")

# --- Run A* with Euclidean Distance ---
start_time = time.time()
a_star_path_euclidean = find_path_using_a_star(maze, start, goal, distance_func=euclidean_distance)
a_star_time_euclidean = time.time() - start_time
visualize_maze_with_path(maze, a_star_path_euclidean, start, goal, seed_value, num_obstacles, "A*", "Euclidean")
print(f"A* with Euclidean Distance Processing Time: {a_star_time_euclidean:.4f} seconds")

# --- Run A* with Left Bias ---
start_time = time.time()
a_star_path_left_bias = find_path_using_a_star(maze, start, goal, left_bias=True)
a_star_time_left_bias = time.time() - start_time
visualize_maze_with_path(maze, a_star_path_left_bias, start, goal, seed_value, num_obstacles, "A*", "Left Bias")
print(f"A* with Left Bias Processing Time: {a_star_time_left_bias:.4f} seconds")
