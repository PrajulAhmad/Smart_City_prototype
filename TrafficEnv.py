import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TrafficIntersectionEnv(gym.Env):
    """
    A simple 4-way intersection environment for Q-Learning.
    
    - State (Baseline): (cars_NS, cars_EW, current_phase, time_in_phase)
    - State (AI-Powered): (cars_NS, cars_EW, current_phase, time_in_phase, 
                           predicted_NS, predicted_EW)
    """
    
    def __init__(self, predictor=None, max_cars_per_lane=20, max_time_in_phase=60):
        super(TrafficIntersectionEnv, self).__init__()
        
        # --- 1. Setup Predictor ---
        self.predictor = predictor
        self.use_predictor = (self.predictor is not None)
        
        # --- 2. Setup Env Parameters ---
        self.max_cars = max_cars_per_lane
        self.max_time = max_time_in_phase
        
        self.action_space = spaces.Discrete(2)
        
        # --- 3. Setup Observation Space (Dynamic) ---
        if self.use_predictor:
            # 6-part AI-Powered State
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
        else:
            # 4-part Baseline State
            self.observation_space = spaces.MultiDiscrete(
                [
                    self.max_cars + 1,  # cars_NS
                    self.max_cars + 1,  # cars_EW
                    2,                  # current_phase
                    self.max_time + 1   # time_in_phase
                ]
            )

        # --- 4. Setup Simulation Parameters ---
        self.car_arrival_prob = 0.3
        self.cars_cleared_per_step = 2

        # --- 5. Initialize Dummy State Variables ---
        # This is the fix: these MUST be defined here
        self.cars_ns = 0
        self.cars_ew = 0
        self.current_phase = 0
        self.time_in_phase = 0
        
        if self.use_predictor:
            self.predicted_ns = 0
            self.predicted_ew = 0

        # self.state is just a placeholder, reset() will set the real one
        self.state = self.get_obs() 
        
        # --- __init__ MUST NOT return anything ---

    def reset(self, seed=None, options=None):
        """
        Resets the environment to a new starting state.
        This is called at the beginning of each episode.
        """
        super().reset(seed=seed)
        
        # Set real starting values
        self.cars_ns = np.random.randint(0, 5)
        self.cars_ew = np.random.randint(0, 5)
        self.current_phase = np.random.randint(0, 2)
        self.time_in_phase = 0
        
        # Get initial predictions if using predictor
        if self.use_predictor:
            self.predicted_ns, self.predicted_ew = self.predictor.predict(
                self.cars_ns, self.cars_ew
            )
        
        # reset() MUST return the observation and an info dict
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
        wait_penalty = (self.cars_ns + self.cars_ew)
        pass_reward = cars_passed
        reward = pass_reward - wait_penalty
        
        # --- 4. Get New Predictions for Next State ---
        if self.use_predictor:
            self.predicted_ns, self.predicted_ew = self.predictor.predict(
                self.cars_ns, self.cars_ew
            )

        # --- 5. Check for "Done" and update state ---
        truncated = self.time_in_phase > self.max_time
        if truncated:
             self.current_phase = 1 - self.current_phase
             self.time_in_phase = 0
             
        # step() MUST return 5 values
        return self.get_obs(), reward, False, truncated, {}

    def get_obs(self):
        """Helper to return the current state as a tuple."""
        if self.use_predictor:
            # Return 6-part AI-Powered State
            return (self.cars_ns, self.cars_ew, self.current_phase, 
                    self.time_in_phase, self.predicted_ns, self.predicted_ew)
        else:
            # Return 4-part Baseline State
            return (self.cars_ns, self.cars_ew, self.current_phase, self.time_in_phase)
