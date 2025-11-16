import numpy as np

class MockEnergyPredictor:
    """
    A mock class representing a trained FL model that forecasts
    energy demand based on time of day.
    """
    def __init__(self):
        # Create a simple 24-hour demand cycle (low at night, high during day)
        self.demand_cycle = np.sin(np.linspace(0, 2 * np.pi, 24)) * 0.4 + 0.5
        self.demand_cycle = np.clip(self.demand_cycle, 0.1, 1.0) # Scale 0.1 to 1.0
        print("Mock Energy Predictor initialized with 24-hour demand cycle.")

    def predict(self, current_hour):
        """
        Simulates a prediction for the *next* hour's demand.
        Adds a little noise.
        """
        next_hour = (current_hour + 1) % 24
        predicted_demand = self.demand_cycle[next_hour] + np.random.normal(0, 0.05)
        return np.clip(predicted_demand, 0.1, 1.0)
        
    # --- FIX: Removed the get_current_demand() method ---
