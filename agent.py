import numpy as np
from collections import defaultdict

class MyAgent:
    def __init__(self, num_agents: int, state_size: int, action_size: int, lr=0.05, gamma=0.95, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.1):
        self.num_agents = num_agents
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Dynamique : stockage uniquement des états visités
        self.q_table = [defaultdict(lambda: [0] * action_size) for _ in range(num_agents)]

        # Raffiner la discrétisation (plus de précision pour x, y)
        self.state_bins = [10, 10, 5, 5, 3, 3]  # Exemple : 10 buckets pour x, y

    def discretize_state(self, state):
        """Transforme un état complexe en indices discrets pour la Q-table."""
        selected_state = state[:6]  # Sélectionner les dimensions importantes
        discrete_state = tuple(np.round(selected_state, 1))  # Arrondir pour réduire la granularité
        return discrete_state

    def get_action(self, state, evaluation=False):
        """Choisit une action pour chaque agent (epsilon-greedy)."""
        actions = []
        for i in range(self.num_agents):
            discrete_state = self.discretize_state(state[i])  # Discrétiser l'état
            q_values = self.q_table[i][discrete_state]  # Récupérer ou initialiser les Q-valeurs
            
            if not evaluation and np.random.rand() < self.epsilon:
                # Exploration : choisir une action aléatoire
                actions.append(np.random.randint(0, self.action_size))
            else:
                # Exploitation : choisir l'action avec la meilleure Q-valeur
                actions.append(np.argmax(q_values))
        return actions

    def update_policy(self, state, action, reward, next_state, done):
        """Met à jour la politique (Q-learning)."""
        for i in range(self.num_agents):
            discrete_state = self.discretize_state(state[i])
            next_discrete_state = self.discretize_state(next_state[i])

            # Récupérer ou initialiser les Q-valeurs actuelles
            q_values = self.q_table[i][discrete_state]
            next_q_values = self.q_table[i][next_discrete_state]

            # Calcul de la mise à jour Q-Learning
            max_next_q = max(next_q_values) if not done else 0
            q_values[action[i]] += self.lr * (reward[i] + self.gamma * max_next_q - q_values[action[i]])

        # Décroissance de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
