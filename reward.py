import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, 
                   evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        # Agent déjà évacué
        if i in evacuated_agents:
            continue

        # Agent désactivé (collision)
        elif i in deactivated_agents:
            rewards[i] = -100.0

        # Agent atteint la cible
        elif tuple(new_pos) in goal_area:
            rewards[i] = 1000.0
            evacuated_agents.add(i)

        else:
            # Récompense/distance
            old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal_area[i]))
            new_dist = np.linalg.norm(np.array(new_pos) - np.array(goal_area[i]))
            dist_diff = old_dist - new_dist

            if dist_diff > 0:
                # Bonus : fixe + proportionnel
                rewards[i] += 2.0 + 1.0 * dist_diff
            elif dist_diff < 0:
                # Pénalité proportionnelle si éloignement
                rewards[i] -= 0.5 * abs(dist_diff)
            else:
                # Légère pénalité si pas de progrès
                rewards[i] -= 0.2

            # Pénalité si l'agent n’a pas bougé du tout
            if np.array_equal(old_pos, new_pos):
                rewards[i] -= 0.2

            # Pénalité fixe par pas de temps
            rewards[i] -= 0.1

    return rewards, evacuated_agents
