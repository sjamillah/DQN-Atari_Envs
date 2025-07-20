
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

 Hyperparameter Set | Noted Behavior |
|--------------------|----------------|
| lr=1e-3, gamma=0.99, batch=32, ε_start=1.0, ε_end=0.1, ε_decay=0.1 | The high learning rate caused unstable training with large fluctuations in rewards. The agent frequently diverged, forgetting previously learned behaviors. Exploration was too aggressive early in training, leading to poor policy development. |
| lr=3e-4, gamma=0.995, batch=64, ε_start=1.0, ε_end=0.05, ε_decay=0.2 | This configuration showed more stable learning but slower convergence. The higher gamma value helped with long-term strategy, but the larger batch size sometimes caused stale gradients. Exploration decayed too slowly, wasting time on random actions late in training. |
| lr=1e-4, gamma=0.99, batch=32, ε_start=1.0, ε_end=0.1, ε_decay=0.1 | Our best configuration showed excellent balance. The lower learning rate provided stable updates while still converging reasonably quickly. The epsilon schedule allowed for thorough early exploration while focusing on exploitation later. Batch size of 32 proved optimal for our hardware constraints. |

Additional observations:
- Gamma values above 0.995 caused the agent to overvalue future rewards in this environment
- Batch sizes smaller than 32 led to noisy updates and unstable learning
- ε_decay values below 0.1 caused the agent to stop exploring too early

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

## Team Collaboration
Our team of 3 members contributed as follows:
Member 1:
Member 2:
Member 3:

## How to Run

### Training
```bash
python train.py
``````
This will:

1. Compare MLP and CNN policies
2. Perform hyperparameter tuning
3. Train final model
4. Save model as dqn_model.zip
5. Generate training curves in results/

### Playing
```bash
python play.py
```

Options:
- --episodes: Number of episodes to run (default: 5)
- --render: Render mode (default: 'human')
- --model_path: Path to trained model (default: 'dqn_model.zip')
