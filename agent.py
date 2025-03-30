import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from env import MazeEnv
from prioritized_replay import PrioritizedReplayBuffer

# Accélération PyTorch
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)


# ----------------------
# 1) DUELING DQN WITH ATTENTION MECHANISM
# ----------------------
class DuelingDQN(nn.Module):
    """
    Dueling network amélioré avec un mécanisme d'attention pour mieux
    traiter les informations des obstacles et autres agents.
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),  # Dropout pour réduire l'overfitting
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
        state_dim=42,                # dimensions de l'état existant
        alpha=1e-4,               # learning rate
        gamma=0.99,                  # discount
        epsilon=1.0,                 # exploration initiale
        epsilon_min=0.05,            # exploration min
        epsilon_decay=0.995,         # vitesse de décroissance de l'exploration
        buffer_size=100000,          # was 50000, increased
        batch_size=128,              # was 128, increased
        detection_radius=5,          # was 3, increased
        update_target_frequency=100, # fréquence de maj du target network
        use_prioritized_replay=True, # Enable prioritized replay
        multi_step=4,                # was 3, increased
        clip_rewards=False,          # (3) Reward clipping toggle
        param_noise=True,            # new
        tau=0.01,                    # new: soft update parameter
        checkpoint_dim=42,
        obstacle_memory_size=10      # NEW: remember obstacles
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.detection_radius = detection_radius

        self.update_target_frequency = update_target_frequency
        self.step_count = 0

        self.use_prioritized_replay = use_prioritized_replay
        if self.use_prioritized_replay:
            self.prioritized_buffer = PrioritizedReplayBuffer(capacity=buffer_size, alpha=0.7)
        else:
            self.replay_buffer = deque(maxlen=buffer_size)
        self.multi_step = multi_step
        self.clip_rewards = clip_rewards
        self.nstep_queue = []  # For storing transitions in multi-step
        self.param_noise = param_noise

        # Utilisation du GPU si possible
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Réseau principal (Dueling DQN)
        self.model = DuelingDQN(
            input_dim=state_dim + 4,
            output_dim=7              
        ).to(self.device)
        
        # Réseau cible
        self.target_model = DuelingDQN(
            input_dim=state_dim + 4,
            output_dim=7
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        # Optimiseur (Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha, weight_decay=1e-5)
        # Loss
        self.criterion = nn.SmoothL1Loss()  # was nn.MSELoss()

        # Pour un entraînement plus stable en FP16 (optionnel)
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.desired_dim = checkpoint_dim
        self.tau = tau
        
        # NEW: memory of dynamic obstacles for better prediction
        self.obstacle_memory = {}
        self.obstacle_memory_size = obstacle_memory_size

    # ----------------------
    # 2.1) Préparation des états
    # ----------------------
    def state_to_tensor(self, states, env_map):
        """
        Convertit la liste d'états en tenseur PyTorch.
        Chaque état s'enrichit de 4 valeurs de détection d'obstacles (detect_obstacles).
        Handles states of different dimensions by padding if necessary.
        """
        if len(states) == 0:
            # Aucun état reçu
            return torch.empty((0, self.desired_dim+4), dtype=torch.float32, device=self.device)

        augmented_states = []
        for s in states:
            s_array = np.array(s, dtype=np.float32).flatten()
            
            # Handle dimension mismatches by padding or truncating
            if len(s_array) != self.desired_dim:
                if len(s_array) < self.desired_dim:
                    # Pad with zeros if state is smaller than expected
                    padded = np.zeros(self.desired_dim, dtype=np.float32)
                    padded[:len(s_array)] = s_array
                    s_array = padded
                else:
                    # Truncate if state is larger than expected
                    s_array = s_array[:self.desired_dim]
            
            obstacle_info = self.detect_obstacles(s_array, env_map)
            augmented_states.append(np.concatenate([s_array, obstacle_info]))

        return torch.tensor(augmented_states, dtype=torch.float32, device=self.device)

    # ----------------------
    # 2.2) Sélection d'actions
    # ----------------------
    def get_action(self, state, env_map=None, evaluation=False, param_noise=False):
        """
        state: un array (num_agents, state_dim)
        On suppose que state_dim >= 12 pour contenir la partie LIDAR (3 directions * 2 infos).
        """
        # Convertir en tenseur pour le DQN, etc.
        if env_map is None:
            # En cas d'évaluation sans env_map, on crée une matrice vide
            env_map = np.zeros((30, 30), dtype=np.int8)
            
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

        # Param noise: shift model weights slightly for exploration
        if (not evaluation) and (param_noise or self.param_noise):
            with torch.no_grad():
                for param in self.model.parameters():
                    param += torch.randn_like(param) * 0.01

        corrected_actions = []
        for i, single_state in enumerate(state):
            # Si l'agent est désactivé ou a atteint son objectif, on ne fait rien
            if single_state[3] > 0:  # Evacuated or deactivated
                corrected_actions.append(0)  # Stay in place
                continue

            # Admettons dist1, type1 = LIDAR principal
            dist_main = single_state[6]
            type_main = single_state[7]
            
            # Si l'action choisie est "avancer" (ex: action=1 = Up) et que dist_main < 1
            # ou type_main est un mur, on choisit autre chose.
            chosen_action = actions[i]
            
            # Évitement intelligent des obstacles
            if chosen_action == 1:  # Move forward
                # Vérifier si on va percuter un obstacle
                if dist_main < 1 and type_main > 0:  # Obstacle detected
                    # Trouver une direction libre
                    if single_state[8] > 1 or single_state[10] > 1:  # Check left and right LIDAR
                        # Si les deux directions sont bloquées, tourner
                        if random.random() < 0.5:
                            chosen_action = 5  # Rotate right
                        else:
                            chosen_action = 6  # Rotate left
                    elif single_state[8] > single_state[10]:  # Left is more open than right
                        chosen_action = 6  # Rotate left
                    else:
                        chosen_action = 5  # Rotate right
                
                # Vérifier les obstacles dynamiques proches
                if self.predict_dynamic_obstacle_collision(single_state, env_map):
                    # Prédiction de collision, éviter
                    if random.random() < 0.5:
                        chosen_action = 5  # Rotate right
                    else:
                        chosen_action = 6  # Rotate left
            
            corrected_actions.append(chosen_action)

        return corrected_actions

    # ----------------------
    # 2.3) NEW: Prédiction des mouvements des obstacles dynamiques
    # ----------------------
    def update_obstacle_memory(self, env_map):
        """
        Met à jour la mémoire des obstacles dynamiques pour suivre leurs mouvements
        """
        # Extraire les positions des obstacles dynamiques (valeur 3 dans la grille)
        current_obstacles = set()
        for x in range(env_map.shape[0]):
            for y in range(env_map.shape[1]):
                if env_map[x, y] == 3:  # Dynamic obstacle
                    current_obstacles.add((x, y))
        
        # Associer les obstacles actuels avec ceux en mémoire (par distance)
        if not self.obstacle_memory:
            # Premier appel, initialiser la mémoire
            for pos in current_obstacles:
                self.obstacle_memory[pos] = deque(maxlen=self.obstacle_memory_size)
                self.obstacle_memory[pos].append(pos)
        else:
            # Mise à jour, associer les obstacles par proximité
            new_memory = {}
            for new_pos in current_obstacles:
                closest_old = None
                min_dist = float('inf')
                
                for old_pos in self.obstacle_memory:
                    dist = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
                    if dist < min_dist and dist < 3:  # Maximum moving distance
                        min_dist = dist
                        closest_old = old_pos
                
                if closest_old:
                    # Transférer l'historique
                    new_memory[new_pos] = self.obstacle_memory[closest_old]
                    new_memory[new_pos].append(new_pos)
                else:
                    # Nouvel obstacle
                    new_memory[new_pos] = deque(maxlen=self.obstacle_memory_size)
                    new_memory[new_pos].append(new_pos)
                    
            self.obstacle_memory = new_memory

    def predict_dynamic_obstacle_collision(self, state, env_map):
        """
        Prédit si un agent risque de rencontrer un obstacle dynamique
        """
        if len(self.obstacle_memory) == 0:
            self.update_obstacle_memory(env_map)
            return False  # Premier appel, pas assez d'info
            
        agent_pos = (state[0], state[1])
        agent_orientation = int(state[2])
        
        # Direction de déplacement selon l'orientation
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        dx, dy = directions[agent_orientation]
        
        # Position future possible de l'agent
        future_agent_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
        
        # Vérifier si un obstacle dynamique pourrait se déplacer vers cette position
        for obs_pos, history in self.obstacle_memory.items():
            if len(history) < 2:
                continue  # Pas assez d'historique
                
            # Calculer le vecteur de déplacement moyen
            if len(history) >= 2:
                last_pos = history[-1]
                prev_pos = history[-2]
                move_dx = last_pos[0] - prev_pos[0]
                move_dy = last_pos[1] - prev_pos[1]
                
                # Position future possible de l'obstacle
                future_obs_pos = (last_pos[0] + move_dx, last_pos[1] + move_dy)
                
                # Vérifier collision
                if future_obs_pos == future_agent_pos:
                    return True
                
                # Vérifier proximité
                dist = np.sqrt((future_agent_pos[0] - future_obs_pos[0])**2 + 
                               (future_agent_pos[1] - future_obs_pos[1])**2)
                if dist < 2:  # Risque élevé
                    return True
        
        return False

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
        # Using multi-step returns if multi_step > 1
        if self.multi_step > 1:
            self.nstep_queue.append((state, action, reward, next_state))
            if len(self.nstep_queue) < self.multi_step:
                return
            # Combine multi-step transitions
            R = 0
            for idx, exp in enumerate(self.nstep_queue):
                R += (self.gamma ** idx) * exp[2]
            s0, a0 = self.nstep_queue[0][0], self.nstep_queue[0][1]
            s_next = next_state
            # Clear the earliest once combined
            self.nstep_queue.pop(0)
            store_tuple = (s0, a0, R, s_next)
        else:
            store_tuple = (state, action, reward, next_state)

        if self.use_prioritized_replay:
            self.prioritized_buffer.add(store_tuple)
        else:
            self.replay_buffer.append(store_tuple)

    # ----------------------
    # 2.5) Entraînement
    # ----------------------
    def train_model(self, env_map):
        """
        Entraîne le DQN sur un batch échantillonné du replay buffer.
        On applique Double DQN + Dueling.
        """
        # Mettre à jour la mémoire des obstacles dynamiques
        self.update_obstacle_memory(env_map)
        
        if self.use_prioritized_replay:
            # Add check to ensure we have enough samples before attempting to sample
            if len(self.prioritized_buffer) < self.batch_size:
                return
            indices, batch, weights = self.prioritized_buffer.sample(self.batch_size, beta=0.5)
            if len(batch) < 1:
                return
        else:
            if len(self.replay_buffer) < self.batch_size:
                return
            batch = random.sample(self.replay_buffer, self.batch_size)
            weights = [1.0]*len(batch)

        # 1) Echantillonner un batch
        # Décomposer (s, a, r, s')
        states_batch = [exp[0] for exp in batch]
        actions_batch = [exp[1] for exp in batch]
        rewards_batch = [exp[2] for exp in batch]
        next_states_batch = [exp[3] for exp in batch]

        if self.clip_rewards:
            rewards_batch = [max(min(r, 1.0), -1.0) for r in rewards_batch]

        # 2) Conversion en tenseurs
        state_tensor = self.state_to_tensor(states_batch, env_map)
        next_state_tensor = self.state_to_tensor(next_states_batch, env_map)
        
        reward_tensor = torch.tensor(rewards_batch, dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(actions_batch, dtype=torch.long, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

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
            batch_indices = torch.arange(len(batch), device=self.device)
            
            # Calcul de la perte
            q_values = current_q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            loss = (self.criterion(q_values, target) * weights_tensor).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        # Clip pour éviter l'explosion des gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # If using prioritized replay, update priorities
        if self.use_prioritized_replay:
            with torch.no_grad():
                td_errors = torch.abs(q_values - target).cpu().numpy()
            self.prioritized_buffer.update_priorities(indices, td_errors)

        # 3) Mise à jour de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 4) Mise à jour du réseau cible
        self.step_count += 1
        if self.step_count % self.update_target_frequency == 0:
            for target_param, main_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)

    # ----------------------
    # 2.6) Passage en mode évaluation
    # ----------------------
    def set_evaluation_mode(self, eval_epsilon=0.0):
        """
        Met le modèle en mode évaluation.
        Paramètres :
            eval_epsilon (float): valeur de epsilon à utiliser en évaluation.
                                   
        """
        self.epsilon = eval_epsilon
        self.model.eval()
        self.target_model.eval()

    # ----------------------
    # 2.7) Checkpoint methods
    # ----------------------
    def save_checkpoint(self, path="multi_config_checkpoint.pth"):
        """
        Save the main/target model states, optimizer, epsilon, and step_count,
        matching how 'simulate.py' structures its checkpoint file.
        """
        checkpoint = {
            "model_main": self.model.state_dict(),
            "model_target": self.target_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path="multi_config_checkpoint.pth"):
        """
        Load the main/target model states, optimizer, epsilon, and step_count,
        matching how 'simulate.py' structures its checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_main"])
        self.target_model.load_state_dict(checkpoint["model_target"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.step_count = checkpoint.get("step_count", self.step_count)
