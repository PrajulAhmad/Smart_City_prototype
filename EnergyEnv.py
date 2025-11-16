import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EnergyEnv(gym.Env):
    """
    A Smart City Energy Management environment.
    
    - Agent: A microgrid controller for a building with a battery.
    - State: (hour_of_day, battery_level, [predicted_demand])
    - Action: 
      - 0: Do nothing (let battery charge/discharge as needed)
      - 1: Force charge battery from grid (buy)
      - 2: Force discharge battery to power building (use battery)
    - Reward: Minimize cost.
    """
    
    def __init__(self, predictor=None, max_steps=24 * 7): # Simulate one week
        super(EnergyEnv, self).__init__()
        
        self.predictor = predictor
        self.use_predictor = (self.predictor is not None)
        self.max_steps = max_steps
        
        # 3 actions: Do nothing, Force Charge, Force Discharge
        self.action_space = spaces.Discrete(3)
        
        # State: (hour_of_day, battery_level_discrete, [predicted_demand_discrete])
        self.battery_capacity = 10.0 # arbitrary units
        self.max_charge_rate = 2.0   # units per hour
        
        # --- FIX: Environment creates its own demand cycle ---
        self.demand_cycle = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.5
        self.demand_cycle = np.clip(self.demand_cycle, 0.1, 1.0)
        # ---
        
        # Initialize state components
        self.current_hour = 0
        self.battery_level = 0.0
        self.predicted_demand = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_hour = np.random.randint(0, 24) # Start at a random time
        self.battery_level = np.random.rand() * self.battery_capacity # 0-10
        self.current_step = 0
        
        if self.use_predictor:
            self.predicted_demand = self.predictor.predict(self.current_hour)
            
        return self.get_obs(), {}

    def get_price(self, hour):
        """Simulates variable electricity prices (cheaper at night)."""
        if 4 <= hour <= 7 or 19 <= hour <= 22:
            return 0.8 # Peak price
        else:
            return 0.2 # Off-peak price

    # --- FIX: New helper method for current demand ---
    def get_current_demand(self, current_hour):
        """Gets the *actual* demand for the current hour."""
        current_demand = self.demand_cycle[current_hour] + np.random.normal(0, 0.05)
        return np.clip(current_demand, 0.1, 1.0)
    # ---

    def step(self, action):
        
        # --- 1. Get current demand and price ---
        current_demand = self.get_current_demand(self.current_hour)
        price = self.get_price(self.current_hour)
        
        cost = 0.0
        
        # --- 2. Apply Action ---
        if action == 1: # Force charge from grid
            charge_amount = self.max_charge_rate
            self.battery_level = min(self.battery_level + charge_amount, self.battery_capacity)
            cost += charge_amount * price # Cost to charge
            
        elif action == 2: # Force discharge to meet demand
            discharged_amount = min(self.battery_level, current_demand)
            self.battery_level -= discharged_amount
            demand_met_from_battery = discharged_amount
            # Buy remaining demand from grid
            remaining_demand = current_demand - demand_met_from_battery
            cost += remaining_demand * price
            
        else: # Action 0: Do nothing (default behavior)
            # Buy all demand from grid
            cost += current_demand * price

        # Discourage letting battery get too full or too empty (add penalty)
        if self.battery_level <= 0.1 or self.battery_level >= self.battery_capacity - 0.1:
            cost += 0.5 # Small penalty for hitting limits
            
        # Reward is negative cost (we want to maximize reward, so minimize cost)
        reward = -cost
        
        # --- 3. Update State ---
        self.current_hour = (self.current_hour + 1) % 24
        
        # Get new predictions
        if self.use_predictor:
            self.predicted_demand = self.predictor.predict(self.current_hour)
            
        # --- 4. Check for Done/Truncated ---
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        # --- FIX: Added the missing return statement ---
        return self.get_obs(), reward, False, truncated, {}

    # --- FIX: Added the missing get_obs() method ---
    def get_obs(self):
        """
        Returns the state as a tuple. We must discretize
        levels for the Q-table.
        """
        # Discretize battery (0-10)
        discrete_battery = int(np.floor((self.battery_level / self.battery_capacity) * 10))
        
        if self.use_predictor:
            # Discretize prediction (0-10)
            discrete_pred = int(np.floor(self.predicted_demand * 10))
            return (self.current_hour, discrete_battery, discrete_pred)
        else:
            return (self.current_hour, discrete_battery)
