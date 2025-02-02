# Solution Overview

## Introduction

This document provides an overview of the solution developed for the hackathon project "Reinforcement Learning for Drone Navigation". The primary focus is on the agent's implementation and the multi-configuration training process, including checkpoint management.

## Approach

### Agent Implementation

The agent is implemented using a Dueling Deep Q-Network (Dueling DQN) architecture. This approach allows the agent to learn both the value of being in a particular state and the advantage of taking a specific action in that state. The agent's architecture and training process are designed to handle the complexities of navigating a grid environment with dynamic obstacles.

#### Key Components

1. **Dueling DQN**: The network consists of a feature extractor, a value stream, and an advantage stream. The Q-values are computed by combining the value and advantage streams.

2. **State Augmentation**: The agent's state is augmented with additional information about nearby obstacles detected using a LIDAR-like sensor. This helps the agent make more informed decisions.

3. **Epsilon-Greedy Exploration**: The agent uses an epsilon-greedy strategy to balance exploration and exploitation. The epsilon value decays over time to reduce exploration as the agent learns.

4. **Experience Replay**: The agent stores experiences in a replay buffer and samples batches of experiences for training. This helps stabilize training by breaking the correlation between consecutive experiences.

5. **Double DQN**: The agent uses Double DQN to reduce overestimation bias by using the target network to select the best action and the main network to evaluate it.

6. **Checkpoint Management**: The agent's state, including the model weights, epsilon value, and step count, is periodically saved to a checkpoint file. This allows training to resume from the last saved state in case of interruptions.

### Multi-Configuration Training

The training process involves training the agent on multiple configurations to make it more generalizable. The configurations are shuffled to avoid learning biases from simpler configurations.

#### Key Steps

1. **Configuration Loading**: Multiple configurations are loaded from JSON files and shuffled.

2. **Environment and Agent Initialization**: The environment and agent are initialized using the first configuration. The agent is optionally loaded from a checkpoint if available.

3. **Training Loop**: The agent is trained across multiple episodes, with configurations randomly selected for each episode. The agent's experiences are stored, and the model is trained using batches from the replay buffer.

4. **Checkpoint Saving**: The agent's state is periodically saved to a checkpoint file to ensure progress is not lost.

5. **Evaluation**: The trained agent is evaluated on multiple configurations to assess its performance. Metrics such as total reward, evacuated agents, and deactivated agents are recorded.

## Code Modifications

### `simulate.py`

- **Function `simulation_config`**: Configures the environment and agent using a JSON configuration file. It also handles loading checkpoints if needed.
- **Function `_load_checkpoint`**: Loads the agent's state from a checkpoint file.
- **Function `multi_config_train`**: Trains the agent on multiple configurations to improve generalization. It periodically saves checkpoints.
- **Function `save_checkpoint`**: Saves the agent's state to a checkpoint file.
- **Function `evaluate`**: Evaluates the trained agent on multiple configurations and records metrics.
- **Function `plot_cumulated_rewards`**: Plots and saves the rewards over episodes.

### `agent.py`

- **Class `DuelingDQN`**: Implements the Dueling DQN architecture.
- **Class `MyAgent`**: Implements the agent's logic, including state preparation, action selection, obstacle detection, experience storage, and model training.
- **Function `train_parallel`**: (Optional) Example code for parallel training using multiple environments.

### `reward.py`

- **Function `compute_reward`**: Computes the reward for each agent based on their actions and positions.

## Conclusion

The solution leverages a Dueling DQN architecture and a robust training process to teach drones to navigate a grid environment with dynamic obstacles. The use of multi-configuration training and checkpoint management ensures the agent is well-trained and resilient to interruptions.

## Additional Notes

- **State Augmentation**: The agent's state is augmented with additional information about nearby obstacles detected using a LIDAR-like sensor. This helps the agent make more informed decisions.
- **Epsilon-Greedy Exploration**: The agent uses an epsilon-greedy strategy to balance exploration and exploitation. The epsilon value decays over time to reduce exploration as the agent learns.
- **Experience Replay**: The agent stores experiences in a replay buffer and samples batches of experiences for training. This helps stabilize training by breaking the correlation between consecutive experiences.
- **Double DQN**: The agent uses Double DQN to reduce overestimation bias by using the target network to select the best action and the main network to evaluate it.
- **Checkpoint Management**: The agent's state, including the model weights, epsilon value, and step count, is periodically saved to a checkpoint file. This allows training to resume from the last saved state in case of interruptions.

By following this approach, the agent is able to learn effective navigation strategies in a complex and dynamic environment, making it a robust solution for the hackathon challenge.
