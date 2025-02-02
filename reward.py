import numpy as np

def compute_reward(num_agents, old_positions, agent_positions, 
                   evacuated_agents, deactivated_agents, goal_area):
    rewards = np.zeros(num_agents)

    for i, (old_pos, new_pos) in enumerate(zip(old_positions, agent_positions)):
        if i in evacuated_agents:
            continue  # Pas de pénalité si déjà évacué

        elif i in deactivated_agents:
            rewards[i] = -50.0  # Moins sévère que -100 pour éviter un blocage complet

        elif tuple(new_pos) in goal_area:
            rewards[i] = 1000.0  # Objectif atteint 🎯
            evacuated_agents.add(i)

        else:
            # Distance avant/après
            old_dist = np.linalg.norm(np.array(old_pos) - np.array(goal_area[i]))
            new_dist = np.linalg.norm(np.array(new_pos) - np.array(goal_area[i]))

            # ✅ **Récompense un meilleur déplacement**
            if new_dist < old_dist:
                rewards[i] += 5.0  # Encouragement + fort pour avancer  
            else:
                rewards[i] -= 1.0  # Pénalité plus faible pour éviter un effet "blocage"

            # 🔄 **Bonus pour un agent qui évite un mur**
            if new_dist == old_dist:  # Cas où l'agent n’a pas pu avancer
                rewards[i] -= 0.2  # Légère punition pour éviter l'immobilité

            # 🔴 **Réduction de la pénalité par step**
            rewards[i] -= 0.05  # Juste pour éviter qu'il reste trop longtemps

    return rewards, evacuated_agents
