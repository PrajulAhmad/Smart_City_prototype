import numpy as np
import random
import json


class QLearningAgent:

    def save_q_table(self, filename):
        """Saves the Q-table to a file."""
        # Convert tuple keys to strings for JSON
        q_table_str_keys = {str(k): v for k, v in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(q_table_str_keys, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, filename):
        """Loads the Q-table from a file."""
        with open(filename, 'r') as f:
            q_table_str_keys = json.load(f)
            # Convert string keys back to tuples
            # This eval is safe because we control the file format
            self.q_table = {eval(k): v for k, v in q_table_str_keys.items()}
        print(f"Q-table loaded from {filename}")
        self.epsilon = 0.0  # Set to exploitation mode

    def __init__(
        self,
        action_space,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        min_epsilon=0.01
    ):
        self.action_space = action_space
        self.q_table = {}  # Use dictionary for sparse states

        # Hyperparameters
        self.alpha = alpha           # Learning rate
        self.gamma = gamma           # Discount factor
        self.epsilon = epsilon       # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair, default to 0 if not seen."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """Choose action using epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            # Explore: choose a random action
            return self.action_space.sample()
        else:
            # Exploit: choose the best action from Q-table
            q_values = [self.get_q_value(state, a) for a in range(self.action_space.n)]
            return np.argmax(q_values)

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using the Bellman equation."""

        # 1. Find max Q-value for the next state
        next_q_values = [self.get_q_value(next_state, a) for a in range(self.action_space.n)]
        max_next_q = max(next_q_values)

        # 2. Current Q-value
        current_q = self.get_q_value(state, action)

        # 3. New Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # 4. Update Q-table
        self.q_table[(state, action)] = new_q

    def update_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

