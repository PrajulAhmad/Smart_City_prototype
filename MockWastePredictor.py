import numpy as np

class MockWastePredictor:
    """
    A mock class representing a trained FL model that predicts
    future waste bin fill levels.
    """
    def __init__(self, num_bins):
        self.num_bins = num_bins
        # Each bin has a different "fill speed"
        self.fill_rates = np.random.rand(self.num_bins) * 0.1 + 0.05 # slow to fast
        print(f"Mock Waste Predictor initialized for {num_bins} bins.")

    def predict(self, current_levels):
        """
        Simulates a prediction for 5 steps (e.g., 5 hours) into the future.
        """
        # A real model would use time-series data.
        # We'll simulate by adding the fill rate 5 times.
        predicted_levels = np.array(current_levels)
        
        # Simulate 5 steps of filling
        for _ in range(5): 
            predicted_levels += self.fill_rates
            
        # Cap at 1.0 (100% full)
        predicted_levels = np.clip(predicted_levels, 0, 1.0)
        
        return predicted_levels
