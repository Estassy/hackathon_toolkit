import numpy as np
import random
from collections import deque
from tensorflow.keras import layers, models, optimizers

class MyAgent:
    def __init__(self, num_agents: int, grid_size: int, 
                 max_lidar_dist_main: int, max_lidar_dist_second: int):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.max_lidar_main = max_lidar_dist_main
        self.max_lidar_second = max_lidar_dist_second
        
        # Configuration de l'état selon env.py
        self.state_size = 42  # 3(pos) + 3(status) + 6(lidar) + 12(agents) + 18(lidar agents)
        self.action_size = 7  # 7 actions possibles

        # Hyperparamètres optimisés
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.batch_size = 64
        self.memory = deque(maxlen=2000)

        # Modèle léger pour entraînement rapide
        self.model = self._build_light_model()
        self.last_states = []

    def _build_light_model(self):
        """Réseau neuronal optimisé pour apprentissage rapide"""
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.005))
        return model

    def get_action(self, state: list, evaluation: bool = False):
        """Sélection d'action hybride RL + règles de sécurité"""
        actions = []
        self.last_states = []
        
        for agent_state in state:
            processed_state = self._preprocess_state(agent_state)
            self.last_states.append(processed_state)
            
            # Règles de sécurité prioritaires
            if self._emergency_avoidance_needed(agent_state):
                action = self._emergency_avoidance(agent_state)
            else:
                # Exploration vs exploitation
                if not evaluation and np.random.rand() <= self.epsilon:
                    action = random.randrange(self.action_size)
                else:
                    q_values = self.model.predict(processed_state, verbose=0)
                    q_values[0][self._goal_direction(agent_state)] += 1.5  # Biais vers le but
                    action = np.argmax(q_values[0])
            
            actions.append(action)
        
        return actions

    def _preprocess_state(self, state):
        """Normalisation des données d'entrée"""
        processed = np.array(state, dtype=np.float32)
        processed[0] /= self.grid_size   # Position X
        processed[1] /= self.grid_size   # Position Y
        processed[2] /= 4                # Orientation
        processed[6] /= self.max_lidar_main   # LIDAR principal
        processed[8] /= self.max_lidar_second # LIDAR droit
        processed[10] /= self.max_lidar_second # LIDAR gauche
        return processed.reshape(1, -1)

    def _emergency_avoidance_needed(self, state):
        """Détection d'obstacle imminent"""
        return (state[6] < 1.5 and state[7] in [1, 2]) or (state[8] < 0.8 and state[9] == 2) or (state[10] < 0.8 and state[11] == 2)        

    def _emergency_avoidance(self, state):
        """Stratégie d'évitement d'urgence"""
        if state[8] > state[10]:  # Plus d'espace à droite
            return 5 if state[2] in [0, 2] else 3  # Rotation droite ou gauche
        else:
            return 6 if state[2] in [1, 3] else 4

    def _goal_direction(self, state):
        """Calcule la direction optimale vers le but"""
        dx = state[4] - state[0]  # goal_x - current_x
        dy = state[5] - state[1]  # goal_y - current_y
        
        if abs(dx) > abs(dy):
            return 3 if dx < 0 else 4  # gauche/droite
        else:
            return 1 if dy < 0 else 2  # haut/bas

    def update_policy(self, actions: list, next_state: list, reward: list):
        """Mise à jour du modèle avec expérience replay"""
        for i in range(self.num_agents):
            if reward[i] == -100:  # Ignorer les agents désactivés
                continue
                
            done = (reward[i] == 1000)
            next_state_processed = self._preprocess_state(next_state[i])
            
            self.memory.append((
                self.last_states[i],
                actions[i],
                reward[i],
                next_state_processed,
                done
            ))

        if len(self.memory) >= self.batch_size:
            self._experience_replay()

    def _experience_replay(self):
        """Apprentissage sur batch d'expériences"""
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = []
        targets = []
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            
            target_vec = self.model.predict(state, verbose=0)
            target_vec[0][action] = target
            
            states.append(state[0])
            targets.append(target_vec[0])
        
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        self.model.save(f'{filename}.h5')

    def load_model(self, filename):
        self.model = models.load_model(f'{filename}.h5')