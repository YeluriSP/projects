import json
import numpy as np

class PredatorPreyEnv:
    def __init__(self, grid, start, monster_start, goal):
        self.grid = grid             # 2D list, each cell has [up,down,left,right] (1=open,0=wall)
        self.m = len(grid)
        self.n = len(grid[0])
        self.start = tuple(start)         # hero start coordinates
        self.monster_start = tuple(monster_start)  # monster start
        self.goal = tuple(goal)           # goal coordinates
        self.reset()

    def reset(self):
        """Reset agents to start positions."""
        self.hero_pos = self.start
        self.mon_pos = self.monster_start
        self.done = False

        # Track visited cells to penalize looping
        self.visited_hero = {self.hero_pos}
        self.visited_mon = {self.mon_pos}

        return self._get_hero_obs(), self._get_mon_obs()

    def _get_hero_obs(self):
        hx, hy = self.hero_pos
        mx, my = self.mon_pos
        gx, gy = self.goal

        wall_up, wall_down, wall_left, wall_right = self.grid[hx][hy]

        return np.array([
            hx, hy,
            mx, my,
            gx, gy,
            wall_up, wall_down, wall_left, wall_right
        ], dtype=np.float32)

    def _get_mon_obs(self):
        hx, hy = self.hero_pos
        mx, my = self.mon_pos

        wall_up, wall_down, wall_left, wall_right = self.grid[mx][my]

        return np.array([
            mx, my,
            hx, hy,
            wall_up, wall_down, wall_left, wall_right
        ], dtype=np.float32)

    def step(self, action_hero, action_mon):
        """
        Perform one simultaneous step. 
        Actions: 0=up,1=down,2=left,3=right.
        Returns: (next_hero_obs, next_mon_obs, hero_reward, mon_reward, done).
        """
        if self.done:
            raise ValueError("Episode is done; call reset().")

        def move(pos, action):
            x, y = pos
            if action == 0 and self.grid[x][y][0]:  # up
                return (x - 1, y)
            if action == 1 and self.grid[x][y][1]:  # down
                return (x + 1, y)
            if action == 2 and self.grid[x][y][2]:  # left
                return (x, y - 1)
            if action == 3 and self.grid[x][y][3]:  # right
                return (x, y + 1)
            return (x, y)  # no move if blocked

        hero_old = self.hero_pos
        mon_old = self.mon_pos

        self.hero_pos = move(hero_old, action_hero)
        self.mon_pos = move(mon_old, action_mon)

        hero_reward = 0.0
        mon_reward = 0.0
        done = False

        if self.mon_pos == self.hero_pos or (self.hero_pos == mon_old and self.mon_pos == hero_old):
            mon_reward += 100.0
            hero_reward -= 50.0
            done = True

        if not done and self.hero_pos == self.goal:
            hero_reward += 200.0
            done = True

        if not done:
            gx, gy = self.goal
            hx_old, hy_old = hero_old
            hx_new, hy_new = self.hero_pos
            old_dist = abs(hx_old - gx) + abs(hy_old - gy)
            new_dist = abs(hx_new - gx) + abs(hy_new - gy)

            episode_scale = getattr(self, "episode_scale", 0.0)
            explore_start = 10.0   # large early bonus
            explore_min = 0.5     # minimum long-term bonus

            explore_bonus = max(explore_min, explore_start * (1.0 - episode_scale))
            base_revisit_penalty = 0.4 + 0.6 * episode_scale  # in [0.4, 1.0]

            # -------- HERO --------
            if self.hero_pos == hero_old:
                hero_reward -= 0.3

            if new_dist < old_dist:
                hero_reward += 0.6
            if not hasattr(self, "visit_counts_hero"):
                self.visit_counts_hero = {}
            self.visit_counts_hero[self.hero_pos] = self.visit_counts_hero.get(self.hero_pos, 0) + 1
            hcount = self.visit_counts_hero[self.hero_pos]

            if hcount == 1:
                hero_reward += explore_bonus
            elif hcount % 3 == 0:
                hero_reward -= base_revisit_penalty * 2.0
            else:
                hero_reward -= base_revisit_penalty * 0.5

            hero_reward -= 0.01

            # -------- MONSTER --------
            mx_old, my_old = mon_old
            mx_new, my_new = self.mon_pos
            old_dist_m = abs(mx_old - hx_old) + abs(my_old - hy_old)
            new_dist_m = abs(mx_new - hx_new) + abs(my_new - hy_new)

            if self.mon_pos == mon_old:
                mon_reward -= 0.25

            if new_dist_m < old_dist_m:
                mon_reward += 0.6

            if not hasattr(self, "visit_counts_mon"):
                self.visit_counts_mon = {}
            self.visit_counts_mon[self.mon_pos] = self.visit_counts_mon.get(self.mon_pos, 0) + 1
            mcount = self.visit_counts_mon[self.mon_pos]

            if mcount == 1:
                mon_explore_bonus = max(0.5, 2.0 * (1.0 - episode_scale))
                mon_reward += mon_explore_bonus
            elif mcount % 3 == 0:
                mon_reward -= (0.3 + 0.5 * episode_scale) * 2.0
            else:
                mon_reward -= (0.3 + 0.5 * episode_scale)

        self.done = done
        return self._get_hero_obs(), self._get_mon_obs(), hero_reward, mon_reward, done
