import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import Dict
from env import MazeEnv
from agent import MyAgent
import random
import torch
import os


def simulation_config(
    config_input,
    new_agent: bool = True,
    checkpoint_path: str = "multi_config_checkpoint.pth",
    load_checkpoint: bool = False,
    use_best_checkpoint: bool = False  # New parameter
):
    """
    Configure the environment and (optionally) an agent using a JSON configuration file.
    Automatically detect the observation dimension from env.reset(), and load checkpoint if needed.
    
    Args:
        config_input: Path to JSON file or config dictionary
        new_agent: Whether to create a new agent
        checkpoint_path: Path to the checkpoint file
        load_checkpoint: Whether to load a checkpoint
        use_best_checkpoint: Whether to use the best checkpoint instead of the regular one
    """

    # 1) Vérifier si l'entrée est un chemin JSON ou un dictionnaire déjà chargé
    if isinstance(config_input, str):  # Cas où on passe un chemin JSON
        with open(config_input, 'r') as config_file:
            config = json.load(config_file)
    elif isinstance(config_input, dict):  # Cas où on passe déjà un dictionnaire
        config = config_input
    else:
        raise ValueError("🚨 ERREUR: config_input doit être un chemin JSON (str) ou un dictionnaire (dict).")


    # 2) Créer l'environnement
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

    # 3) Obtenir un dummy state pour détecter la taille d'observation
    dummy_state, _ = env.reset()
    # dummy_state.shape = (num_agents, obs_dim) si l'env est multi-agents
    obs_dim = dummy_state.shape[1] if dummy_state.shape[0] > 0 else 0

    # 4) Créer l'agent si nécessaire
    agent = None
    if new_agent:
        agent = MyAgent(
            num_agents=config.get('num_agents'),  # Only loads regular checkpoint, never the "_best" one
            state_dim=obs_dim,  # ou obs_dim + 4 selon tes besoins
            # d'autres hyperparams si besoin
        )

        # 5) Charger un checkpoint si présent
        if load_checkpoint:
            if use_best_checkpoint:
                # Try to load the best checkpoint first
                best_path = checkpoint_path.replace('.pth', '_best.pth')
                if os.path.isfile(best_path):
                    success = _load_checkpoint(agent, best_path)
                    if success:
                        print(f"Meilleur checkpoint chargé avec succès!")
                    else:
                        # Fall back to regular checkpoint if best fails
                        _load_checkpoint(agent, checkpoint_path)
                else:
                    # Fall back to regular checkpoint if best doesn't exist
                    _load_checkpoint(agent, checkpoint_path)
            else:
                # Just load the regular checkpoint
                _load_checkpoint(agent, checkpoint_path)

    return env, agent, config


def _load_checkpoint(agent: MyAgent, ckpt_path: str):
    """
    Tente de charger un checkpoint (modèle,optimizer, epsilon, etc.) si le fichier existe.
    En cas d'erreur de dimension, on ignore le checkpoint et on repart sur des poids aléatoires.
    
    Returns:
        bool: True if checkpoint was loaded successfully, False otherwise
    """
    if not os.path.isfile(ckpt_path):
        print(f"Fichier de checkpoint non trouvé: {ckpt_path}")
        return False

    print(f"Tentative de chargement du checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=agent.device)
        agent.model.load_state_dict(checkpoint["model_main"])
        agent.target_model.load_state_dict(checkpoint["model_target"])

        # Charger l'état de l'optimiseur (important pour reprendre l'entraînement là où il était)
        if "optimizer_state" in checkpoint:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state"])
            
        agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
        agent.step_count = checkpoint.get("step_count", agent.step_count)
        print(f"Checkpoint chargé avec succès depuis {ckpt_path} !")
        return True
    except RuntimeError as e:
        print(f"ERREUR : {e}")
        print("Mismatch de dimension ? On ignore ce checkpoint.")
        return False


def multi_config_train(
    config_paths, 
    max_total_episodes=3000, 
    checkpoint_path="multi_config_checkpoint.pth", 
    save_interval=100,
    validation_interval=300  # New parameter
):
    """
    Entraîne un agent sur plusieurs configurations JSON pour le rendre plus généraliste.
    
    Args:
        config_paths (list): Liste des chemins des fichiers JSON.
        max_total_episodes (int): Nombre total d'épisodes pour l'entraînement.
        checkpoint_path (str): Emplacement où sauvegarder le modèle entraîné.
        save_interval (int): Intervalle d'épisodes pour sauvegarder un checkpoint.
        validation_interval (int): Intervalle d'épisodes pour valider le modèle.
    """
    print("\n🚀 Début de l'entraînement multi-configurations 🚀\n")
    
    # Charger toutes les configurations
    configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            configs.append(json.load(f))
    
    # Mélange les configurations pour éviter d'apprendre uniquement sur les plus simples
    random.shuffle(configs)
    
    # Initialisation de l'environnement et de l'agent avec la première config
    env, agent, _ = simulation_config(config_paths[0], new_agent=True, checkpoint_path=checkpoint_path, load_checkpoint=True)
    
    # Vérification si agent est bien initialisé
    if agent is None:
        print("🚨 ERREUR : L'agent n'a pas été initialisé correctement par simulation_config()")
        return None, None
    
    all_rewards = []
    episode_count = 0
    best_reward = float('-inf')
    
    try:
        while episode_count < max_total_episodes:
            # Sélectionner une configuration aléatoire
            config = random.choice(configs)  # Prendre une config depuis la liste déjà mélangée
            env, _, _ = simulation_config(config, new_agent=False)


            # Vérifier si l'agent est bien créé
            if agent is None:
                print("🚨 ERREUR : L'agent n'a pas été initialisé correctement par simulation_config()")
                return None, None
            
            state, info = env.reset()
            total_reward = 0.0
            terminated, truncated = False, False
            
            while not (terminated or truncated):
                env_map = env.grid
                actions = agent.get_action(state, env_map, evaluation=False)
                next_state, rewards, terminated, truncated, info = env.step(actions)
                
                total_reward += np.sum(rewards)
                
                # Fix: Use the actual number of agents in the current state
                current_num_agents = len(state)
                for i in range(min(agent.num_agents, current_num_agents)):
                    agent.store_experience(state[i], actions[i], rewards[i], next_state[i])
                
                agent.train_model(env_map)
                state = next_state
            
            all_rewards.append(total_reward)
            print(f"Épisode {episode_count+1}/{max_total_episodes}, Reward: {total_reward:.2f}")
            
            # Sauvegarde périodique du modèle (correcte)
            if (episode_count + 1) % save_interval == 0:
                save_checkpoint(agent, checkpoint_path) 
                print(f"💾 Checkpoint sauvegardé (épisode {episode_count+1}).")
            
            # Validation périodique sur un environnement différent
            if (episode_count + 1) % validation_interval == 0:
                validation_reward = validate_agent(agent, random.choice(configs))
                print(f"📊 Validation (épisode {episode_count+1}): Reward = {validation_reward:.2f}")
            
            # Sauvegarde du meilleur modèle
            best_reward = save_best_checkpoint(agent, total_reward, best_reward, checkpoint_path)
            
            episode_count += 1
            
    except KeyboardInterrupt:
        print("🛑 Entraînement interrompu par l'utilisateur.")
    
    finally:
        env.close()
    
    # Sauvegarde finale du modèle
    save_checkpoint(agent, checkpoint_path)
    print("✅ Entraînement terminé. Modèle sauvegardé sous", checkpoint_path)
    
    return agent, all_rewards


def save_checkpoint(agent: MyAgent, ckpt_path: str):
    """
    Sauvegarde le modèle principal + target, epsilon, step_count, etc. dans un seul .pth
    """
    checkpoint = {
        "model_main": agent.model.state_dict(),
        "model_target": agent.target_model.state_dict(),
        "optimizer_state": agent.optimizer.state_dict(),
        "epsilon": agent.epsilon,
        "step_count": agent.step_count
    }
    torch.save(checkpoint, ckpt_path)


def save_best_checkpoint(agent: MyAgent, reward: float, best_reward: float, ckpt_path: str) -> float:
    """
    Sauvegarde le checkpoint uniquement si la récompense actuelle est meilleure que la meilleure précédente.
    Retourne la meilleure récompense (mise à jour si nécessaire).
    """
    if reward > best_reward:
        best_path = ckpt_path.replace('.pth', '_best.pth')
        save_checkpoint(agent, best_path)
        return reward
    return best_reward


def validate_agent(agent: MyAgent, config: dict) -> float:
    """
    Validate the agent on a given configuration and return the total reward.
    """
    env, _, _ = simulation_config(config, new_agent=False)
    state, info = env.reset()
    total_reward = 0.0
    terminated, truncated = False, False

    while not (terminated or truncated):
        actions = agent.get_action(state, evaluation=True)
        state, rewards, terminated, truncated, info = env.step(actions)
        total_reward += np.sum(rewards)

    env.close()
    return total_reward


def evaluate(configs_paths: list, trained_agent: MyAgent, num_episodes: int = 10, use_best_checkpoint: bool = True) -> pd.DataFrame:
    """
    Evaluate a trained agent on multiple configurations, calculate metrics, and visualize results.
    Reverted to the older evaluation approach.
    """

    import pandas as pd
    import time

    all_results = pd.DataFrame()

    for config_path in configs_paths:
        print(f"\n--- Evaluating Configuration: {config_path} ---")

        env, _, config = simulation_config(config_path, new_agent=False, use_best_checkpoint=use_best_checkpoint)

        metrics = []
        total_reward = 0
        episode_count = 0

        state, info = env.reset()
        time.sleep(1)

        try:
            while episode_count < num_episodes:
                actions = trained_agent.get_action(state, evaluation=True)
                state, rewards, terminated, truncated, info = env.step(actions)
                total_reward += sum(rewards)
                
                print(f"\rEpisode {episode_count + 1}/{num_episodes}, Step {info['current_step']}, "
                    f"Reward: {total_reward:.2f}, "
                    f"Evacuated: {len(info['evacuated_agents'])}, "
                    f"Deactivated: {len(info['deactivated_agents'])}", end='')

                time.sleep(1)

                if terminated or truncated:
                    print("\r")
                    metrics.append({
                        "config_path": config_path,
                        "episode": episode_count + 1,
                        "steps": info['current_step'],
                        "reward": total_reward,
                        "evacuated": len(info['evacuated_agents']),
                        "deactivated": len(info['deactivated_agents'])
                    })

                    episode_count += 1
                    total_reward = 0

                    if episode_count < num_episodes:
                        state, info = env.reset()

        except KeyboardInterrupt:
            print("\nSimulation interrupted by the user")

        finally:
            env.close()

        config_results = pd.DataFrame(metrics)
        all_results = pd.concat([all_results, config_results], ignore_index=True)

    all_results.to_csv('all_results.csv', index=False)
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
