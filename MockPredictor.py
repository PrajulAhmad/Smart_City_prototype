import numpy as np

class MockTrafficPredictor:
    """
    A mock class representing our trained Federated Learning (FL) LSTM model.
    In a real implementation, this class would load the trained model
    (e.g., a .h5 file) and call model.predict().
    
    For this simulation, it just returns a simple, plausible "prediction"
    based on current traffic.
    """
    def __init__(self, prediction_horizon=5):
        self.horizon = prediction_horizon # e.g., predicts 5 steps (minutes) ahead
        print("Mock FL Predictor initialized.")

    def predict(self, current_cars_ns, current_cars_ew):
        """
        Simulates a prediction.
        
        Logic: Predicts that a certain fraction of the current cars will
        remain, and adds a small random "new arrival" amount.
        """
        # A real model would take a time-series history, but we'll
        # use the current state for this mock-up.
        
        predicted_ns = int(current_cars_ns * 0.5 + np.random.randint(0, 3))
        predicted_ew = int(current_cars_ew * 0.5 + np.random.randint(0, 3))
        
        # Ensure predictions are within a reasonable bound
        max_cars = 20 # Should match the env's max_cars
        predicted_ns = min(predicted_ns, max_cars)
        predicted_ew = min(predicted_ew, max_cars)

        return predicted_ns, predicted_ew
