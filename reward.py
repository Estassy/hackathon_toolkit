import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, 
                   evacuated_agents, deactivated_agents, goal_area):
    """
    - Si l'agent est désactivé : -100
    - Si l'agent atteint la zone but : +1000
    - Step penalty global : -0.1 par step
    - Bonus pour la réduction de distance vers le but
    - Légère pénalité si l'agent ne bouge pas
    """
    rewards = np.zeros(num_agents, dtype=np.float32)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        # Agent déjà évacué
        if i in evacuated_agents:
            continue

        # Collision => désactivé
        elif i in deactivated_agents:
            rewards[i] = -100.0

        # L'agent est dans la zone but
        elif tuple(new_pos) in goal_area:
            rewards[i] = 1000.0
            evacuated_agents.add(i)

        else:
            # Pénalité par step
            rewards[i] -= 0.1

            # Calcul de la distance au but avant/après
            old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal_area[i]))
            new_dist = np.linalg.norm(np.array(new_pos) - np.array(goal_area[i]))

            # Récompense pour la réduction de distance
            dist_diff = old_dist - new_dist
            if dist_diff > 0:
                # On peut ajuster le coefficient en fonction de la taille de la grille
                rewards[i] += 1.0 * dist_diff  

            # Pénalité légère si l'agent n'a pas bougé
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 0.1

    return rewards, evacuated_agents
