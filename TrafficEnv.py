import gymnasium as gym
from gymnasium import spaces
import numpy as np
# We don't need to import the predictor here, 
# it will be passed in during __init__

class TrafficIntersectionEnv(gym.Env):
    """
    An AI-Powered 4-way intersection environment.
    
    - State: (cars_NS, cars_EW, current_phase, time_in_phase, 
              predicted_NS, predicted_EW)
    - Actions: 0 (Stay), 1 (Switch)
    """
    
    def __init__(self, predictor, max_cars_per_lane=20, max_time_in_phase=60):
        super(TrafficIntersectionEnv, self).__init__()
        
        self.predictor = predictor # <-- NEW: Store the predictor
        self.max_cars = max_cars_per_lane
        self.max_time = max_time_in_phase
        
        # Define Action Space (Discrete: 0 or 1)
        self.action_space = spaces.Discrete(2)
        
        # --- MODIFIED: Observation Space ---
        # Now has 6 elements:
        # (cars_NS, cars_EW, current_phase, time_in_phase, predicted_NS, predicted_EW)
        self.observation_space = spaces.MultiDiscrete(
            [
                self.max_cars + 1,  # cars_NS
                self.max_cars + 1,  # cars_EW
                2,                  # current_phase
                self.max_time + 1,  # time_in_phase
                self.max_cars + 1,  # predicted_NS
                self.max_cars + 1   # predicted_EW
            ]
        )
        
        # Simulation parameters
        self.car_arrival_prob = 0.3
        self.cars_cleared_per_step = 2

        self.state = (0, 0, 0, 0, 0, 0) # Initial dummy state

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        
        self.cars_ns = np.random.randint(0, 5)
        self.cars_ew = np.random.randint(0, 5)
        self.current_phase = np.random.randint(0, 2)
        self.time_in_phase = 0
        
        # Get initial predictions
        self.predicted_ns, self.predicted_ew = self.predictor.predict(
            self.cars_ns, self.cars_ew
        )
        
        self.state = (self.cars_ns, self.cars_ew, self.current_phase, 
                      self.time_in_phase, self.predicted_ns, self.predicted_ew)
        
        return self.get_obs(), {}

    def step(self, action):
        """Execute one time step in the environment."""
        
        # --- 1. Apply Action ---
        if action == 1: # Switch phase
            self.current_phase = 1 - self.current_phase
            self.time_in_phase = 0
        else: # Stay on phase
            self.time_in_phase += 1
            
        # --- 2. Update Environment (Simulate Traffic) ---
        cars_passed = 0
        if self.current_phase == 0: # NS is green
            cars_passed = min(self.cars_ns, self.cars_cleared_per_step)
            self.cars_ns -= cars_passed
        else: # EW is green
            cars_passed = min(self.cars_ew, self.cars_cleared_per_step)
            self.cars_ew -= cars_passed
            
        # Add new arriving cars
        if np.random.rand() < self.car_arrival_prob:
            self.cars_ns = min(self.cars_ns + 1, self.max_cars)
        if np.random.rand() < self.car_arrival_prob:
            self.cars_ew = min(self.cars_ew + 1, self.max_cars)
            
        # --- 3. Calculate Reward ---
        # We can make the reward smarter now, but for simplicity,
        # we'll keep it the same. The agent will learn to use
        # the new state info to optimize this reward.
        wait_penalty = (self.cars_ns + self.cars_ew)
        pass_reward = cars_passed
        reward = pass_reward - wait_penalty
        
        # --- 4. Get New Predictions for Next State ---
        self.predicted_ns, self.predicted_ew = self.predictor.predict(
            self.cars_ns, self.cars_ew
        )

        # --- 5. Check for "Done" and update state ---
        truncated = self.time_in_phase > self.max_time
        if truncated:
             self.current_phase = 1 - self.current_phase
             self.time_in_phase = 0
             
        # --- MODIFIED: Update the full 6-part state ---
        self.state = (self.cars_ns, self.cars_ew, self.current_phase, 
                      self.time_in_phase, self.predicted_ns, self.predicted_ew)
        
        return self.get_obs(), reward, False, truncated, {}

    def get_obs(self):
        """Helper to return the current state as a tuple."""
        return self.state
