import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

class CNN_DuelingDQN(nn.Module):
    def __init__(self, n_actions, h, w):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        conv_out_size = 64 * h * w

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU()
        )

        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, n_actions)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        val = self.value(x)
        adv = self.advantage(x)
        return val + adv - adv.mean(dim=1, keepdim=True)


class DQNAgent:
    def __init__(self, n_actions, grid_h, grid_w, lr=1e-3, gamma=0.99):

        self.gamma = gamma
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.net = CNN_DuelingDQN(n_actions, grid_h, grid_w).to(self.device)
        self.target_net = CNN_DuelingDQN(n_actions, grid_h, grid_w).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = collections.deque(maxlen=50000)

        self.learn_step = 0
        self.target_update_freq = 300

    def act(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.n_actions)

        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.net(state_v)

        return int(q.argmax(dim=1).item())

    def remember(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self, batch):

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_actions = self.net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def replay(self, batch_size=64):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        self.train_step(batch)
