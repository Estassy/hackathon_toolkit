import numpy as np

class MyAgent():
    def __init__(self, num_agents: int, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):        
        # Parameters for Q-learning
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Random number generator for random actions pick
        self.rng = np.random.default_rng()
        self.num_agents = num_agents

        # Q-table for storing state-action values (initialized as a dictionary)
        # Keys: state-action pairs, Values: Q-values
        self.q_table = {}

    def state_to_tuple(self, state):
        """Convert the state representation into a hashable tuple."""
        return tuple(map(tuple, state))  # Assuming state is a nested list or array

    def get_action(self, state: list, evaluation: bool = False):
        """Choose an action using an epsilon-greedy policy."""
        state_tuple = self.state_to_tuple(state)

        actions = []
        for _ in range(self.num_agents):
            if not evaluation and self.rng.random() < self.epsilon:
                # Explore: Choose a random action
                action = self.rng.integers(0, 7)
            else:
                # Exploit: Choose the best action based on Q-values
                q_values = self.q_table.get(state_tuple, np.zeros(7))  # Default Q-values are 0
                action = np.argmax(q_values)
            actions.append(action)

        return actions

    def update_policy(self, actions: list, state: list, reward: list, next_state: list):
        """Update Q-values based on the Q-learning formula."""
        state_tuple = self.state_to_tuple(state)
        next_state_tuple = self.state_to_tuple(next_state)

        for agent_idx, action in enumerate(actions):
            # Get current Q-value for the state-action pair
            current_q_value = self.q_table.get(state_tuple, np.zeros(7))[action]

            # Get the maximum Q-value for the next state
            max_next_q_value = np.max(self.q_table.get(next_state_tuple, np.zeros(7)))

            # Update Q-value
            updated_q_value = current_q_value + self.alpha * (
                reward[agent_idx] + self.gamma * max_next_q_value - current_q_value
            )

            # Save updated Q-value in the Q-table
            if state_tuple not in self.q_table:
                self.q_table[state_tuple] = np.zeros(7)
            self.q_table[state_tuple][action] = updated_q_value

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# def update_policy(self, actions: list, state: list, reward: list, next_state: list):
#     """Update Q-values based on the Double Q-learning formula."""
#     state_tuple = self.state_to_tuple(state)
#     next_state_tuple = self.state_to_tuple(next_state)

#     for agent_idx, action in enumerate(actions):
#         # Get current Q-value for the state-action pair
#         current_q_value = self.q_table.get(state_tuple, np.zeros(7))[action]

#         # Get the action with the highest Q-value for the next state
#         best_next_action = np.argmax(self.q_table.get(next_state_tuple, np.zeros(7)))
        
#         # Get the Q-value for the best action in the next state from the target Q-table
#         max_next_q_value = self.target_q_table.get(next_state_tuple, np.zeros(7))[best_next_action]

#         # Update Q-value
#         updated_q_value = current_q_value + self.alpha * (
#             reward[agent_idx] + self.gamma * max_next_q_value - current_q_value
#         )

#         # Save updated Q-value in the Q-table
#         if state_tuple not in self.q_table:
#             self.q_table[state_tuple] = np.zeros(7)
#         self.q_table[state_tuple][action] = updated_q_value

#     # Update target Q-table periodically
#     if self.epsilon > self.epsilon_min:
#         self.epsilon *= self.epsilon_decay
#     if self.update_counter % self.target_update_freq == 0:
#         self.target_q_table = self.q_table.copy()
#     self.update_counter += 1