import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WasteEnv(gym.Env):
    """
    A Smart City Waste Management environment.
    
    - Agent: A single collection truck.
    - State: (truck_location, bin_levels_... , [predicted_levels_...])
    - Action: `Discrete(num_bins)` -> "Go to bin X"
    - Reward: High for full bins, low for empty bins, penalty for travel.
    """
    
    def __init__(self, num_bins=5, predictor=None, max_steps=50):
        super(WasteEnv, self).__init__()
        
        self.num_bins = num_bins
        self.predictor = predictor
        self.use_predictor = (self.predictor is not None)
        self.max_steps = max_steps
        
        # Action: Go to bin 0, 1, 2, ...
        self.action_space = spaces.Discrete(self.num_bins)
        
        # State: (truck_location, bin1_level, ..., binN_level)
        # We use a simple Box space. We'll flatten the state tuple.
        base_state_size = 1 + self.num_bins
        if self.use_predictor:
            # Add predicted levels
            state_size = base_state_size + self.num_bins
        else:
            state_size = base_state_size
            
        # We need a simple, discrete state for Q-learning.
        # Let's discretize the levels (e.g., 0-10) and truck location.
        # State: (truck_loc, bin1_level_discrete, ..., binN_level_discrete, [pred_...])
        
        # --- For simplicity, we will use a tuple state and let the Q-table handle it ---
        # Note: This is less efficient than a flat Box, but works for Q-learning.
        
        # --- Let's define the state components ---
        self.truck_location = 0
        self.bin_levels = np.zeros(self.num_bins)
        self.predicted_levels = np.zeros(self.num_bins)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start truck at a random bin
        self.truck_location = np.random.randint(0, self.num_bins)
        # Start bins at random (low) fill levels
        self.bin_levels = np.random.rand(self.num_bins) * 0.4
        
        self.current_step = 0
        
        if self.use_predictor:
            self.predicted_levels = self.predictor.predict(self.bin_levels)
            
        return self.get_obs(), {}

    def step(self, action):
        """Action = the bin to travel to."""
        destination_bin = action
        
        # --- 1. Calculate Reward ---
        travel_distance = abs(self.truck_location - destination_bin)
        
        current_fill_level = self.bin_levels[destination_bin]
        
        if current_fill_level > 0.8:
            # Big reward for collecting a full bin
            reward = 20
        elif current_fill_level < 0.2:
            # Big penalty for wasting a trip to an empty bin
            reward = -10
        else:
            # Small reward for collecting a partially-full bin
            reward = 5
            
        # Add travel penalty
        reward -= travel_distance
        
        # --- 2. Update State ---
        # Truck moves to the new location
        self.truck_location = destination_bin
        # Bin is emptied
        self.bin_levels[destination_bin] = 0.0
        
        # All other bins fill up a little bit
        self.bin_levels += np.random.rand(self.num_bins) * 0.02
        self.bin_levels = np.clip(self.bin_levels, 0, 1.0)
        
        # Get new predictions
        if self.use_predictor:
            self.predicted_levels = self.predictor.predict(self.bin_levels)
            
        # --- 3. Check for Done/Truncated ---
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        return self.get_obs(), reward, False, truncated, {}

    def get_obs(self):
        """
        Returns the state as a tuple. We must discretize
        levels for the Q-table.
        """
        # Discretize levels (0-10)
        discrete_levels = tuple(np.floor(self.bin_levels * 10).astype(int))
        
        if self.use_predictor:
            discrete_preds = tuple(np.floor(self.predicted_levels * 10).astype(int))
            return (self.truck_location,) + discrete_levels + discrete_preds
        else:
            return (self.truck_location,) + discrete_levels
