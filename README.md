
# Deep Q-Learning for Atari Boxing

## Project Overview
This project implements a Deep Q-Network (DQN) agent to play the Atari Boxing game using Stable Baselines3 and Gymnasium. The solution includes:
- A training script (`train.py`) that compares MLP vs CNN policies and performs hyperparameter tuning
- A playing script (`play.py`) that loads the trained model and demonstrates gameplay
- Comprehensive documentation of the training process and results

## Implementation

### Understanding of DQN and RL Concepts
Our implementation demonstrates:
- Proper handling of exploration-exploitation tradeoff using ε-greedy policy
- Experience replay buffer for training stability
- Target network implementation to prevent oscillation
- Frame stacking to capture temporal information

### Hyperparameter Tuning
We tested multiple configurations and documented results:

| Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| lr=1e-3, gamma=0.99, batch=32, ε_start=1.0, ε_end=0.1, ε_decay=0.1 | High learning rate caused unstable training |
| lr=3e-4, gamma=0.995, batch=64, ε_start=1.0, ε_end=0.05, ε_decay=0.2 | Better stability but slower convergence |
| lr=1e-4, gamma=0.99, batch=32, ε_start=1.0, ε_end=0.1, ε_decay=0.1 | Best balance of stability and performance |

### Training Process
The training script (`train.py`) performs:
1. Environment setup with frame stacking
2. Policy comparison (MLP vs CNN)
3. Hyperparameter optimization
4. Final model training with evaluation callbacks
5. Model saving and performance visualization

Key training parameters:
- Total timesteps: 500,000
- Frame stack: 4
- Buffer size: 50,000
- Target network update: Every 2,500 steps

### Evaluation and Agent Performance
The play script (`play.py`) provides:
- Model loading from saved file
- Environment rendering with human-readable display
- Greedy policy evaluation (no exploration)
- Performance statistics collection
- Optional video recording

## Play Demo
