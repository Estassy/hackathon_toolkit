

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from typing import Tuple, Optional, Dict

from env import MazeEnv
from agent import MyAgent

def simulation_config(config_path: str, new_agent: bool = True) -> Tuple[MazeEnv, Optional[MyAgent], Dict]:
    """
    Configure the environment and optionally an agent using a JSON configuration file.

    Args:
        config_path (str): Path to the configuration JSON file.
        new_agent (bool): Whether to initialize the agent. Defaults to True.

    Returns:
        Tuple[MazeEnv, Optional[MyAgent], Dict]: Configured environment, agent (if new), and the configuration dictionary.
    """
    
    # Read config
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    # Env configuration
    env = MazeEnv(
        size=config.get('grid_size'),
        walls_proportion=config.get('walls_proportion'),
        num_dynamic_obstacles=config.get('num_dynamic_obstacles'),
        num_agents=config.get('num_agents'),
        communication_range=config.get('communication_range'),
        max_lidar_dist_main=config.get('max_lidar_dist_main'),
        max_lidar_dist_second=config.get('max_lidar_dist_second'),
        max_episode_steps=config.get('max_episode_steps'),
        render_mode=config.get('render_mode', None),
        seed=config.get('seed', None)
    )

    # Agent configuration
    agent = MyAgent(num_agents=config.get('num_agents')) if new_agent else None

    return env, agent, config

def train(config_path: str) -> Tuple[MyAgent, list]:
    """
    Train an agent on the configured environment.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        Tuple[MyAgent, list]: The trained agent and the list of rewards per episode.
    """
    env, agent, config = simulation_config(config_path)
    max_episodes = config.get('max_episodes')

    all_rewards = []
    episode_count = 0

    try:
        while episode_count < max_episodes:
            state, info = env.reset()
            total_reward = 0
            terminated = False

            while not terminated:
                # Choose actions based on the current state
                actions = agent.get_action(state)

                # Perform the actions in the environment
                next_state, rewards, terminated, truncated, info = env.step(actions)

                # Update the agent's policy
                agent.update_policy(actions, state, rewards, next_state)

                # Accumulate the total reward for the episode
                total_reward += sum(rewards)

                # Transition to the next state
                state = next_state

                # Optionally render the environment
                if config.get('render_mode') == 'human':
                    time.sleep(0.1)

            # Log the total reward for this episode
            all_rewards.append(total_reward)
            print(f"Episode {episode_count + 1}/{max_episodes}, Total Reward: {total_reward}")

            episode_count += 1

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    finally:
        env.close()

    return agent, all_rewards

def evaluate(configs_paths: list, trained_agent: MyAgent, num_episodes: int = 10) -> pd.DataFrame:
    """
    Evaluate a trained agent on multiple configurations, calculate metrics, and visualize results.

    Args:
        configs_paths (list): List of paths to the configuration JSON files.
        trained_agent (MyAgent): A pre-trained agent to evaluate.
        num_episodes (int): Number of episodes to run for evaluation per configuration. Defaults to 10.

    Returns:
        pd.DataFrame: A DataFrame containing evaluation metrics for each episode and configuration.
    """
    all_results = pd.DataFrame()

    for config_path in configs_paths:
        print(f"\n--- Evaluating Configuration: {config_path} ---")

        env, _, config = simulation_config(config_path, new_agent=False)

        metrics = []

        try:
            for episode in range(num_episodes):
                state, info = env.reset()
                total_reward = 0
                terminated = False

                while not terminated:
                    actions = trained_agent.get_action(state, evaluation=True)
                    state, rewards, terminated, truncated, info = env.step(actions)
                    total_reward += sum(rewards)

                metrics.append({
                    "config_path": config_path,
                    "episode": episode + 1,
                    "total_reward": total_reward,
                    "evacuated_agents": len(info['evacuated_agents']),
                    "deactivated_agents": len(info['deactivated_agents']),
                })

        except KeyboardInterrupt:
            print("Evaluation interrupted by user.")

        finally:
            env.close()

        config_results = pd.DataFrame(metrics)
        all_results = pd.concat([all_results, config_results], ignore_index=True)

    all_results.to_csv('evaluation_results.csv', index=False)
    return all_results

def plot_cumulated_rewards(rewards: list, interval: int = 100):
    """
    Plot and save the rewards over episodes.

    Args:
        rewards (list): List of total rewards per episode.
        interval (int): Interval between ticks on the x-axis (default is 100).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards)+1), rewards, color='blue', marker='o', linestyle='-')
    plt.title('Total Cumulated Rewards per Episode')
    plt.xlabel('Episodes')

    # Adjust x-ticks to display every 'interval' episodes
    xticks = range(1, len(rewards)+1, interval)
    plt.xticks(xticks)

    plt.ylabel('Cumulated Rewards')
    plt.grid(True)
    plt.savefig('reward_curve_per_episode.png', dpi=300)
    plt.show()
