import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.01):
        self.action_space = action_space
        self.q_table = {} # Use a dictionary for sparse states
        
        # Hyperparameters
        self.alpha = alpha       # Learning rate
        self.gamma = gamma       # Discount factor (value of future rewards)
        self.epsilon = epsilon   # Exploration rate
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
        """Update the Q-table using the Bellman equation."""
        
        # Q(s, a) = Q(s, a) + alpha * [reward + gamma * max_a'(Q(s', a')) - Q(s, a)]
        
        # 1. Find max Q-value for the next state
        next_q_values = [self.get_q_value(next_state, a) for a in range(self.action_space.n)]
        max_next_q = max(next_q_values)
        
        # 2. Get the current Q-value
        current_q = self.get_q_value(state, action)
        
        # 3. Calculate the new Q-value
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        # 4. Update the table
        self.q_table[(state, action)] = new_q

    def update_epsilon(self):
        """Decay the exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
