import json
from grid_env import PredatorPreyEnv
from steps_dnn import DQNAgent
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from maze_generator import generate_maze, find_deadends

def draw_maze(ax, grid):
    m, n = len(grid), len(grid[0])
    ax.set_xlim([-0.5, n - 0.5])
    ax.set_ylim([m - 0.5, -0.5])
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw walls
    for i in range(m):
        for j in range(n):
            up, down, left, right = grid[i][j]
            if up == 0:
                ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color='black', lw=2)
            if down == 0:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color='black', lw=2)
            if left == 0:
                ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color='black', lw=2)
            if right == 0:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color='black', lw=2)

# Load maze
with open("maze_10.json") as f:
    data = json.load(f)

m, n = data["grid_size"]
grid_data = [[None for _ in range(n)] for _ in range(m)]
for key, val in data["cells"].items():
    i, j = eval(key)
    grid_data[i][j] = val

env = PredatorPreyEnv(grid_data, data["start"], data["end"], data["goal"])

# Agents
hero_agent = DQNAgent(input_dim=6, n_actions=4, lr=1e-3, gamma=0.9)
mon_agent  = DQNAgent(input_dim=4, n_actions=4, lr=1e-3, gamma=0.9)

# Training loop
num_episodes = 1500
max_steps = 150
eps_start, eps_end = 1.0, 0.1

plt.ion()
fig, ax = plt.subplots()

# Track how many episodes we’ve played on the current maze
wins_on_current_maze = 0  
local_maze_counter = 0

for ep in range(num_episodes):
    env.episode_scale = min(1.0, local_maze_counter / 60.0)
    hero_obs, mon_obs = env.reset()
    total_r_h = 0.0
    total_r_m = 0.0
    eps = max(0.05, 1.0 - ep / 800)
    done = False

    for step in range(max_steps):
        a_h = hero_agent.act(hero_obs, eps)
        weak_prob = max(0.0, 0.5 - ep / 900)
        if random.random() < weak_prob:
            a_m = random.randint(0, 3)
        else:
            a_m = mon_agent.act(mon_obs, eps)
        next_h, next_m, r_h, r_m, done = env.step(a_h, a_m)
        total_r_h += r_h
        total_r_m += r_m

        hero_agent.remember(hero_obs, a_h, r_h, next_h, done)
        mon_agent.remember(mon_obs, a_m, r_m, next_m, done)
        hero_agent.replay(batch_size=64)
        mon_agent.replay(batch_size=64)

        hero_obs, mon_obs = next_h, next_m

        if done:
            wins_on_current_maze += 1
            print(f"Episode {ep+1}: Hero reward={total_r_h:.2f}, Monster reward={total_r_m:.2f}")
            break

        # Visualization
        if ep % 50 == 0:   # show once every 50 episodes
            ax.clear()
            draw_maze(ax, grid_data)
            hx, hy = env.hero_pos; mx, my = env.mon_pos; gx, gy = env.goal
            ax.scatter(hy, hx, c='blue', s=100)
            ax.scatter(my, mx, c='red', s=100, marker='x')
            ax.scatter(gy, gx, c='green', s=100, marker='*')
            plt.pause(0.01)
    local_maze_counter += 1

    # === Maze change logic ===
    if wins_on_current_maze >= 75:
        print("10 terminal episodes completed on this maze. Generating a new one...")
        m, n = env.m, env.n
        new_grid = generate_maze(m, n)
        deadends = find_deadends(new_grid)
        if deadends:
            goal = random.choice(deadends)
        else:
            goal = (m // 2, n // 2)
        env = PredatorPreyEnv(new_grid, [0, 0], [m - 1, n - 1], goal)
        grid_data = new_grid
        wins_on_current_maze = 0
        local_maze_counter =0
        print("New maze generated.")

    # === Model saving ===
    if (ep + 1) % 50 == 0:
        torch.save(hero_agent.net.state_dict(), f"saved_models/hero_dqn_ep{ep+1}.pth")
        torch.save(mon_agent.net.state_dict(), f"saved_models/monster_dqn_ep{ep+1}.pth")
        print("Models saved.")

    print(f"Episode {ep+1} done — Hero reward={total_r_h:.2f}, Monster reward={total_r_m:.2f}")

# plt.ioff()
