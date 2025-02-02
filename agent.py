import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import json

from env import MazeEnv

# Accélération PyTorch
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)


# ----------------------
# 1) DUELING DQN
# ----------------------
class DuelingDQN(nn.Module):
    """
    Dueling network: on calcule la valeur d'état (V) et l'avantage (A) pour chaque action,
    puis Q(s,a) = V(s) + A(s,a) - mean(A(s,*)).
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Stream de valeur (Value)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Sortie = valeur V(s)
        )
        
        # Stream d'avantage (Advantage)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Sortie = avantage A(s,a)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


# ----------------------
# 2) AGENT - MyAgent
# ----------------------
class MyAgent():
    def __init__(
        self,
        num_agents: int,
        state_dim=42,                 # dimensions de l'état existant
        alpha=0.0005,                # learning rate
        gamma=0.99,                  # discount
        epsilon=1.0,                 # exploration initiale
        epsilon_min=0.05,            # exploration min
        epsilon_decay=0.995,         # vitesse de décroissance de l'exploration
        buffer_size=10000,
        batch_size=128,
        detection_radius=3,
        update_target_frequency=100   # fréquence de maj du target network
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_agents = num_agents
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.detection_radius = detection_radius

        self.update_target_frequency = update_target_frequency
        self.step_count = 0

        # Utilisation du GPU si possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseau principal (Dueling DQN)
        self.model = DuelingDQN(
            input_dim=state_dim + 4,  # +4 pour la détection d'obstacles ou pas
            output_dim=7              # 7 actions
        ).to(self.device)
        
        # Réseau cible
        self.target_model = DuelingDQN(
            input_dim=state_dim + 4,
            output_dim=7
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimiseur (Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        # Loss
        self.criterion = nn.MSELoss()

        # Pour un entraînement plus stable en FP16 (optionnel)
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    # ----------------------
    # 2.1) Préparation des états
    # ----------------------
    def state_to_tensor(self, states, env_map):
        """
        Convertit la liste d'états en tenseur PyTorch.
        Chaque état s'enrichit de 4 valeurs de détection d'obstacles (detect_obstacles).
        """
        if len(states) == 0:
            # Aucun état reçu
            return torch.empty((0, 42+4), dtype=torch.float32, device=self.device)

        augmented_states = []
        for s in states:
            s = np.array(s, dtype=np.float32).flatten()
            obstacle_info = self.detect_obstacles(s, env_map)
            augmented_states.append(np.concatenate([s, obstacle_info]))

        return torch.tensor(augmented_states, dtype=torch.float32, device=self.device)

    # ----------------------
    # 2.2) Sélection d'actions
    # ----------------------
    def get_action(self, state, env_map, evaluation=False):
        """
        state: un array (num_agents, state_dim)
        On suppose que state_dim >= 12 pour contenir la partie LIDAR (3 directions * 2 infos).
        """
        # Convertir en tenseur pour le DQN, etc.
        state_tensor = self.state_to_tensor(state, env_map)

        # Exploration epsilon-greedy
        if (not evaluation) and (random.random() < self.epsilon):
            # Actions brutes issues de l'exploration
            actions = [random.randint(0, 6) for _ in range(state_tensor.shape[0])]
        else:
            # Actions issues du réseau Q
            with torch.no_grad():
                q_values = self.model(state_tensor)
                actions = torch.argmax(q_values, dim=1).tolist()

        # ÉTAPE CLÉ : on va "corriger" l'action si le LIDAR indique un obstacle tout proche
        corrected_actions = []
        for i, single_state in enumerate(state):
            # single_state contient [x, y, orientation, status, goal_x, goal_y,
            #   dist1, type1, dist2, type2, dist3, type3, ...]

            # Admettons dist1,type1 = LIDAR principal
            dist_main = single_state[6]
            type_main = single_state[7]
            
            # Si l'action choisie est "avancer" (ex: action=1 = Up) et que dist_main < 1
            # ou type_main est un mur, on choisit autre chose.
            # Ici, c'est juste un exemple, tu dois l'adapter à ta logique d'actions
            # ou à l'orientation si tu as un "moving forward" dépendant de l'orientation...
            
            chosen_action = actions[i]
            
            # Ex: action=1 (move up) => on suppose que c'est "avancer" dans la direction principale
            # (Si dans ton env, 1=Up absolu, c'est un peu différent.)
            if chosen_action == 1 and dist_main < 1 and type_main == 1:
                # On décide de ne pas avancer, on reste immobile par exemple
                chosen_action = 0  # action=0 => immobile
            
            corrected_actions.append(chosen_action)

        return corrected_actions


    # ----------------------
    # 2.3) Détection locale d'obstacles
    # ----------------------
    def detect_obstacles(self, state, env_map):
        """
        state = (x, y, orientation, etc.)
        On regarde 4 directions (gauche, droite, haut, bas) jusqu'à detection_radius.
        Si on trouve un mur/obstacle dynamique, on renvoie un score pondéré par la distance.
        """
        obstacle_data = np.zeros(4, dtype=np.float32)

        if len(state) < 2:
            return obstacle_data  # Sécurité

        agent_x, agent_y = int(state[0]), int(state[1])

        # directions: [gauche, droite, haut, bas]
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        for i, (dx, dy) in enumerate(directions):
            for step in range(1, self.detection_radius + 1):
                x, y = agent_x + dx*step, agent_y + dy*step
                if 0 <= x < env_map.shape[0] and 0 <= y < env_map.shape[1]:
                    # Mur (1) ou obstacle dynamique (3)
                    if env_map[x, y] == 1 or env_map[x, y] == 3:
                        # Score = 1 - (distance / detection_radius)
                        obstacle_data[i] = 1.0 - (step / self.detection_radius)
                        break
                else:
                    # Hors de la grille => mur ou bloc
                    obstacle_data[i] = 1.0
                    break
        return obstacle_data

    # ----------------------
    # 2.4) Stockage d'expériences
    # ----------------------
    def store_experience(self, state, action, reward, next_state):
        """
        On stocke l'expérience (s, a, r, s') dans le replay buffer pour l'entraînement DQN.
        """
        self.replay_buffer.append((state, action, reward, next_state))

    # ----------------------
    # 2.5) Entraînement
    # ----------------------
    def train_model(self, env_map):
        """
        Entraîne le DQN sur un batch échantillonné du replay buffer.
        On applique Double DQN + Dueling.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # 1) Echantillonner un batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        # Décomposer (s, a, r, s')
        states_batch = [exp[0] for exp in batch]
        actions_batch = [exp[1] for exp in batch]
        rewards_batch = [exp[2] for exp in batch]
        next_states_batch = [exp[3] for exp in batch]

        # 2) Conversion en tenseurs
        state_tensor = self.state_to_tensor(states_batch, env_map)
        next_state_tensor = self.state_to_tensor(next_states_batch, env_map)
        
        reward_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions_batch, dtype=torch.long, device=self.device)

        with torch.amp.autocast(device_type="cuda", enabled=(self.device.type=="cuda")):
            # Q(s, a) courant
            current_q_values = self.model(state_tensor)

            # Double DQN:
            #  - On choisit l'action qui maximise Q dans le réseau principal
            #  - On utilise le réseau cible pour calculer la valeur de cette action
            with torch.no_grad():
                # argmax_a Q_main(next_state, a)
                next_q_main = self.model(next_state_tensor)
                best_actions = torch.argmax(next_q_main, dim=1)

                # Q_target(next_state, a_best)
                next_q_target = self.target_model(next_state_tensor)
                best_future_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            # Calcul de la cible de Q:
            # r + gamma * Q_target(s', a_best)
            target = reward_tensor + (self.gamma * best_future_q)

            # On veut seulement mettre à jour Q(s, a) de l'action jouée
            # => On clone, on remplace la valeur sur l'indice action_tensor
            target_q_values = current_q_values.clone()
            batch_indices = torch.arange(self.batch_size, device=self.device)
            target_q_values[batch_indices, action_tensor] = target

            # Calcul de la perte
            loss = self.criterion(current_q_values, target_q_values)

        # Backprop
        self.scaler.scale(loss).backward()
        # Clip pour éviter explosion des gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # 3) Mise à jour de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 4) Mise à jour du réseau cible
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# ----------------------
# 3) Multiprocessing (optionnel)
# ----------------------
def train_parallel(env_config, agent, episodes=1000):
    """
    Exemple de code pour faire tourner plusieurs environnements en parallèle.
    Si tu ne l'utilises pas, tu peux le laisser commenté.
    """
    num_workers = mp.cpu_count()
    envs = [MazeEnv(**env_config) for _ in range(num_workers)]
    
    def worker(env):
        for _ in range(episodes // num_workers):
            state, _ = env.reset()
            while True:
                actions = agent.get_action(state, env.get_env_map())
                next_state, rewards, done, _, _ = env.step(actions)
                
                # Stockage
                for i in range(agent.num_agents):
                    agent.store_experience(
                        state[i],
                        actions[i],
                        rewards[i],
                        next_state[i]
                    )
                
                # Entraînement
                agent.train_model(env.get_env_map())
                state = next_state
                if done:
                    break

    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(num_workers)
    pool.map(worker, envs)
    pool.close()
    pool.join()
