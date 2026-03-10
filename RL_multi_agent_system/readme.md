# Hero-Monster Reinforcement Learning

A multi-agent reinforcement learning project where a **hero** agent learns to reach a goal while evading a **monster** agent that chases it. Both agents learn simultaneously using Deep Q-Learning (DQN) in dynamically generated maze environments.

## Project Overview

This project implements a predator-prey reinforcement learning environment with two competing agents:
- **Hero (Prey)**: Learns to navigate mazes and reach a goal while avoiding the monster
- **Monster (Predator)**: Learns to chase and catch the hero

The agents learn through three different approaches:
1. **DNN-based**: Full grid state representation
2. **CNN-based**: Grid-as-image state representation (convolutional neural networks)
3. **Local Wall-based**: Partial observability (only local wall information)

## Project Structure

```
hero_monster/
├── train.py                 # Main training loop (DNN approach)
├── train_cnn.py            # Training script for CNN agents
├── train_local_wall.py     # Training script for local wall observation variant
├── grid_env.py             # Environment class (DNN variant)
├── grid_env_cnn.py         # Environment class (CNN variant)
├── grid_env_local_wall.py  # Environment class (local wall variant)
├── steps_dnn.py            # DQN agent implementation (dense layers)
├── steps_cnn.py            # DQN agent implementation (convolutional layers)
├── maze_generator.py       # Procedural maze generation
├── maze_10.json            # Pre-generated 10x10 maze
├── saved_models/           # Saved model weights directory
├── saved_models_cnn/       # Saved CNN model weights directory
├── saved_models_local/     # Saved local-wall model weights directory
└── README.md               # This file
```

## Architecture

### Environment (`grid_env*.py`)
- **State space**: Different representations depending on variant:
  - **DNN**: 6D vector (hero position, monster position, goal position)
  - **CNN**: 4-channel image (walls, hero, monster, goal)
  - **Local Wall**: 10D vector (positions + local wall configuration)
- **Action space**: 4 discrete actions (up, down, left, right)
- **Rewards**:
  - Hero reaches goal: +200
  - Monster catches hero: -50 (hero), +100 (monster)
  - Moving closer to goal: +0.6
  - Exploring new cells: +0.5 to +10 (decreases over training)
  - Revisiting cells: -0.4 to -1.0 (increases over training)

### Agent Architecture (`steps_dnn.py`, `steps_cnn.py`)
- **Network Type**: Dueling DQN
  - Feature extraction → Value stream + Advantage stream
  - Combines value of state with advantage of each action
- **Replay Buffer**: 50,000 transitions
- **Target Network**: Updated every 200-300 steps
- **Optimizer**: Adam with learning rate 1e-3
- **Discount Factor**: γ = 0.9

### Maze Generation (`maze_generator.py`)
- Generates random mazes using Depth-First Search (DFS)
- Adds random loops for complexity
- Automatically places goal at dead-ends
- Configurable grid dimensions

## Getting Started

### Prerequisites
```
python >= 3.8
torch >= 1.9
numpy
matplotlib
json (built-in)
random (built-in)
```

### Installation
```bash
# Install dependencies
pip install torch numpy matplotlib

# Navigate to project directory
cd hero_monster
```

### Running Training

**DNN-based training** (full state observation):
```bash
python train.py
```

**CNN-based training** (image-based observation):
```bash
python train_cnn.py
```

**Local Wall-based training** (partial observability):
```bash
python train_local_wall.py
```

## Training Details

### Key Hyperparameters
- **Episodes**: 1500-5000 (depending on variant)
- **Max steps per episode**: 100-150
- **Batch size**: 64
- **Learning rate**: 0.001
- **Epsilon decay**: Decreases from 1.0 to 0.05 over ~800 episodes

### Curriculum Learning
- **Maze progression**: After 50-75 wins on a maze, a new one is generated
- **Episode scaling**: Exploration bonus decreases as training progresses
- **Monster handicap**: Early episodes may disable monster to help hero learn movement

### Model Checkpoints
Models are saved every 50-100 episodes in the respective `saved_models*` directories.

## Training Objectives

### Hero Agent
- Reach the goal while avoiding the monster
- Learn efficient pathfinding through exploration
- Balance exploration (finding new areas) with exploitation (reaching goal)

### Monster Agent
- Chase and catch the hero
- Learn predict hero movement patterns
- Navigate the maze effectively to intercept the hero

## Variants Explained

| Feature | DNN | CNN | Local Wall |
|---------|-----|-----|-----------|
| State Input | Vector (6D/4D) | Image (4 channels) | Vector (10D) |
| Network | Dense layers | Conv + Dense | Dense layers |
| Observability | Full grid | Full grid (as image) | Local walls only |
| Episodes | 1500 | 5000 | 2500 |
| Maze switch | 75 wins | 50 wins | 75 wins |

## Expected Training Progression

1. **Early training (episodes 0-200)**: Random exploration, agents learn basic movement
2. **Mid training (episodes 200-800)**: Hero learns pathfinding, monster learns pursuit
3. **Late training (episodes 800+)**: Agents become competent, curriculum increases maze difficulty

## Saving & Loading Models

Models are automatically saved using PyTorch's `state_dict()`:
```python
# Saving
torch.save(agent.net.state_dict(), f"saved_models/hero_dqn_ep{ep}.pth")

# Loading
agent.net.load_state_dict(torch.load("saved_models/hero_dqn_ep1000.pth"))
agent.net.eval()
```

## Key Learning Concepts

- **Deep Q-Learning (DQN)**: Value-based reinforcement learning
- **Dueling DQN**: Separates state value from action advantages
- **Experience Replay**: Breaks temporal correlations in experience
- **Target Networks**: Stabilizes training by using separate networks
- **Multi-agent Learning**: Both agents trained simultaneously with competing objectives
- **Curriculum Learning**: Progressive difficulty increase with new maze generation

## Customization

### Modifying Rewards
Edit the reward logic in `grid_env*.py` `step()` method:
```python
hero_reward += 0.6  # Moving closer to goal
mon_reward += 100.0  # Catching hero
```

### Changing Maze Complexity
In `maze_generator.py`, adjust:
```python
m, n = 10, 10  # Grid dimensions
```

### Adjusting Training Parameters
In `train*.py`, modify:
```python
num_episodes = 1500
max_steps = 150
lr = 1e-3
```

## Troubleshooting

**GPU not being used:**
- Ensure PyTorch CUDA version matches your GPU drivers
- Check: `torch.cuda.is_available()`

**Notes:**
- All models use GPU if available, otherwise CPU
- Training time varies based on hardware (GPU recommended)
- Experiments are logged to console with episode rewards
