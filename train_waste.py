import numpy as np
from WasteEnv import WasteEnv
from RLAgent import QLearningAgent
from MockWastePredictor import MockWastePredictor

# --- 1. Setup ---
print("Setting up AI-Powered Waste environment and agent...")
NUM_BINS = 5
predictor = MockWastePredictor(num_bins=NUM_BINS)
env = WasteEnv(num_bins=NUM_BINS, predictor=predictor)
agent = QLearningAgent(action_space=env.action_space, epsilon_decay=0.9999, min_epsilon=0.1)

# --- 2. Training ---
print("Starting RL Training...")
total_steps = 200_000 # Waste mgmt is a bit more complex, train longer
log_interval = 20_000

state, _ = env.reset()
total_reward = 0

for step in range(total_steps):
    action = agent.choose_action(state)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.update_q_table(state, action, reward, next_state)
    
    state = next_state
    agent.update_epsilon()
    total_reward += reward
    
    if (step + 1) % log_interval == 0:
        avg_reward = total_reward / log_interval
        print(f"Steps {step+1-log_interval}-{step+1}: "
              f"Avg Reward = {avg_reward:.2f}, "
              f"Epsilon = {agent.epsilon:.3f}")
        total_reward = 0
        
    if truncated:
        state, _ = env.reset()

print("Training complete.")

# --- 3. Save the Agent ---
agent.save_q_table("ai_waste_agent.json")
print("AI-Powered Waste Agent saved to 'ai_waste_agent.json'")
