import json
import numpy as np

class PredatorPreyEnv:
    def __init__(self, grid, start, monster_start, goal):
        self.grid = grid             
        self.m = len(grid)
        self.n = len(grid[0])
        self.start = tuple(start)    
        self.monster_start = tuple(monster_start) 
        self.goal = tuple(goal)      
        self.reset()

    def _get_cnn_state(self):
        h, w = self.m, self.n
        state = np.zeros((4, h, w), dtype=np.float32)

        for i in range(h):
            for j in range(w):
                if sum(self.grid[i][j]) < 4:   
                    state[0, i, j] = 1.0

        hx, hy = self.hero_pos
        state[1, hx, hy] = 1.0

        mx, my = self.mon_pos
        state[2, mx, my] = 1.0

        gx, gy = self.goal
        state[3, gx, gy] = 1.0

        return state

    def reset(self):
        self.hero_pos = self.start
        self.mon_pos = self.monster_start
        self.done = False

        self.visit_counts_hero = {}
        self.visit_counts_mon = {}

        state = self._get_cnn_state()
        return state, state

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
            return (x, y)  

        # Save old positions
        hero_old = self.hero_pos
        mon_old = self.mon_pos

        # Update positions
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

            explore_start = 10.0  
            explore_min = 0.5     

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
        state = self._get_cnn_state()
        return state, state, hero_reward, mon_reward, done
