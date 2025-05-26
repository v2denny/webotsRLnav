# WebotsRLnav

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Webots](https://img.shields.io/badge/webots-2023a+-orange.svg)](https://cyberbotics.com/)


Reinforcement learning framework for robotic navigation using curriculum learning in Webots simulation environment. Implements and compares PPO, DQN, SAC, and TD3 algorithms for E-Puck robot navigation with obstacle avoidance and multi-target path planning.

## Key Results

Curriculum learning provides **5x faster training** compared to direct training on complex scenarios. Performance comparison after 600k training steps:

| Algorithm | Success Rate | Avg Steps | Collision Rate |
|-----------|--------------|-----------|----------------|
| TD3       | 90.2%        | 31.3      | 9.8%           |
| SAC       | 67.0%        | 38.5      | 33.0%          |
| PPO       | 61.0%        | 32.8      | 39.0%          |
| DQN       | 19.1%        | 232.1     | 80.9%          |

**TD3 with continuous actions emerged as the best performer**, achieving the highest success rate with efficient navigation and minimal collisions.

## Features

- Multiple RL algorithms: PPO, DQN, SAC, TD3
- Curriculum learning implementation
- Discrete and continuous action spaces
- Comprehensive metrics tracking
- TensorBoard integration
- Predictive collision avoidance
- Multi-target navigation support

## Requirements

- Python 3.8+
- Webots R2023a+
- Gymnasium
- Stable-Baselines3
- NumPy
- TensorBoard



## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/WebotsRLnav.git
cd WebotsRLnav
```

2. Create conda environment:
```bash
conda env create -f env.yaml
conda activate webots-rl
```

3. Install dependencies:
```bash
pip install stable-baselines3[extra] tensorboard
```

## Project Structure

```
WebotsRLnav/
├── Report_TRI2.pdf              # Research paper
├── env.yaml                     # Environment configuration
├── environment.py               # Discrete action environment
├── environment_cont.py          # Continuous action environment
├── training.py                  # Training script
├── testing.py                   # Testing script
├── scripts/
│   ├── utils.py                 # Robot utilities
│   └── positions.py             # Map positions
├── worlds/                      # Webots world files
├── models/                      # Trained models
├── metrics/                     # Training metrics
└── logs/                        # TensorBoard logs
```

## Usage

### Training

1. Edit the main section in `training.py`:
```python
if __name__ == "__main__":
    env, model, summary = train_agent(
        algorithm='TD3',                      # PPO, DQN, SAC, TD3
        mode="hard",                         # start, easy, medium, hard, all, random
        total_steps=100000,
        model_path='models/existing_model.zip',  # Optional: continue training
        metrics_prefix=None
    )
```

2. Run training:
```bash
python training.py
```

### Testing

1. Edit the test configuration in `testing.py`:
```python
test_config = {
    'algorithm': 'TD3',                      # PPO, DQN, SAC, TD3
    'mode': 'random',                        # start, easy, medium, hard, all, random
    'model_path': 'models/td3_all_600000.zip',  # Path to trained model
    'num_episodes': 100,                     # Number of test episodes
    'save_folder': 'benchmarking/results'    # Results folder
}
```

2. Run testing:
```bash
python testing.py
```

### Monitoring

Launch TensorBoard to monitor training:
```bash
tensorboard --logdir=./logs/
```

## Curriculum Learning

The training uses a progressive curriculum with 5 stages:

- **START** (25k steps): Single target, clear path, auto-oriented
- **EASY** (50k steps): Multiple positions, single obstacles, auto-oriented  
- **MEDIUM** (125k steps): Easy scenarios with random orientation
- **HARD** (200k steps): Complex obstacles, multi-targets, auto-oriented
- **ALL** (100k steps): All scenarios with random orientation

## Environment Configuration

### Robot Sensors
- LiDAR: 25 rays, 150° FOV
- GPS: Position tracking
- Touch Sensor: Collision detection

### Action Spaces
- **Discrete**: {Forward, Turn Left, Turn Right}
- **Continuous**: [forward_speed, rotation_speed]

### Observation Space (27D)
- 25 LiDAR distance readings [0, 2]
- Distance to target [0, MAX_DISTANCE]
- Angle to target [-π, π]


# Development Notes

## Design Decisions

### Action Configuration
- **Timestep**: 200ms (vs 100ms) - Provides better stability for decision making
- **Action Space**: 3 actions (Forward, Turn Left, Turn Right) - 4th action (backwards) adds unnecessary complexity without benefits

### Algorithm Selection
Tested 4 algorithms across discrete and continuous action spaces:

**Discrete Action Space:**
- PPO - Robust performance with excellent curriculum learning benefits
- DQN - Moderate performance, some curriculum learning benefits

**Continuous Action Space:**
- SAC - Strong exploration with entropy regularization
- TD3 - Best overall performance with twin critic architecture

## Reward Function Evolution

### Initial Issues & Solutions

**Obstacle Avoidance:**
- Problem: Robot didn't fear obstacles enough
- Solution: Gradient penalty based on distance to walls instead of threshold-based

**Target Hovering:**
- Problem: Robot lingered near target instead of completing episodes
- Solution: Increased time penalty, added diminishing returns for extreme proximity

**Wall Stuck Behavior:**
- Problem: Robot oscillated left-right when facing walls
- Solution: Reward consistent rotation direction when wall-stuck

**Corner Navigation:**
- Problem: Getting stuck on corners and long walls
- Solution: Enhanced exploration rewards, improved turning incentives

## Environment Improvements

### Sensor Configuration
- **LiDAR**: Upgraded from 9 to 25 rays for better obstacle detection
- **Range**: Removed ray clipping for full sensor range
- **Positioning**: Improved sensor placement in Webots environment

### Collision Detection
- **Predictive System**: Prevents collisions before they occur
- **Multi-step Verification**: Enhanced detection reliability
- **Action Blocking**: Temporary prevention of risky forward movements

### Safety Features
- Pre-action collision checking
- Predictive collision avoidance with 42° frontal cone analysis
- Adaptive penalty system for repeated risky behavior

## Contact

- Daniel Dias - up202105076@fe.up.pt
- Lucas Santiago - up202104660@fe.up.pt
- Nuno Moreira - up202104873@fe.up.pt
- Rafael Conceição - up202006898@fe.up.pt
