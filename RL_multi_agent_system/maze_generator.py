import random, json

def generate_maze(m, n):
    grid = [[[0, 0, 0, 0] for _ in range(n)] for _ in range(m)]
    visited = [[False] * n for _ in range(m)]
    
    directions = [(-1, 0, 0, 1), (1, 0, 1, 0), (0, -1, 2, 3), (0, 1, 3, 2)]  # (di, dj, wall_current, wall_next)
    
    def dfs(i, j):
        visited[i][j] = True
        dirs = directions[:]
        random.shuffle(dirs)
        for di, dj, w1, w2 in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and not visited[ni][nj]:
                grid[i][j][w1] = 1
                grid[ni][nj][w2] = 1
                dfs(ni, nj)
    
    dfs(0, 0)

    # Add some random extra connections to create loops
    for _ in range(m * n // 4):
        i, j = random.randint(0, m-1), random.randint(0, n-1)
        di, dj, w1, w2 = random.choice(directions)
        ni, nj = i + di, j + dj
        if 0 <= ni < m and 0 <= nj < n:
            grid[i][j][w1] = 1
            grid[ni][nj][w2] = 1

    # Set borders to 0
    for i in range(m):
        grid[i][0][2] = 0
        grid[i][n-1][3] = 0
    for j in range(n):
        grid[0][j][0] = 0
        grid[m-1][j][1] = 0

    return grid


def find_deadends(grid):
    deadends = []
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if sum(grid[i][j]) == 1:
                deadends.append((i, j))
    return deadends


def save_to_json(grid, filename="maze_10.json"):
    m, n = len(grid), len(grid[0])
    data = {
        "grid_size": [m, n],
        "cells": {f"({i},{j})": grid[i][j] for i in range(m) for j in range(n)},
        "start": [0, 0],
        "end": [m - 1, n - 1]
    }
    deadends = find_deadends(grid)
    if deadends:
        data["goal"] = random.choice(deadends)
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Maze saved to {filename}")


# Example usage
# m, n = 10, 10
# maze = generate_maze(m, n)
# save_to_json(maze)