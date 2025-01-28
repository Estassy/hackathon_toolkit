import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue
        elif i in deactivated_agents:
            rewards[i] = -50.0  # Réduit la pénalité pour désactivation
        elif tuple(new_pos) in goal_area:
            rewards[i] = 500.0  # Récompense plus élevée pour atteindre l'objectif
            evacuated_agents.add(i)
        else:
            rewards[i] = -0.05  # Réduit les pénalités pour chaque étape

    return rewards, evacuated_agents

