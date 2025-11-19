# Import our custom classes
from TrafficEnv import TrafficIntersectionEnv
from RLAgent import QLearningAgent
from MockPredictor import MockTrafficPredictor

# --- 1. Setup ---
print("Setting up AI-Powered environment and agent...")

predictor = MockTrafficPredictor()
env = TrafficIntersectionEnv(predictor=predictor) 
agent = QLearningAgent(action_space=env.action_space)

# --- 2. Training ---
print("Starting RL Training...")

# UPDATED training steps
total_steps = 500_000 
log_interval = 25_000 # Increased log interval

state, _ = env.reset()
total_reward = 0
reward_log = []

for step in range(total_steps):
    action = agent.choose_action(state)
    next_state, reward, done, truncated, _ = env.step(action)
    agent.update_q_table(state, action, reward, next_state)
    
    state = next_state
    agent.update_epsilon()
    total_reward += reward
    
    if (step + 1) % log_interval == 0:
        avg_reward = total_reward / log_interval
        reward_log.append(avg_reward)
        print(f"Steps {step+1-log_interval}-{step+1}: "
              f"Avg Reward = {avg_reward:.2f}, "
              f"Epsilon = {agent.epsilon:.3f}")
        total_reward = 0
        
    if truncated:
        state, _ = env.reset()

print("Training complete.")

# --- 3. Test the "Trained" Agent ---
print("\n--- Testing Trained AI-Powered Agent (Epsilon=0) ---")
state, _ = env.reset()
agent.epsilon = 0.0 # Turn off exploration
total_test_reward = 0
total_test_steps = 1000

for _ in range(total_test_steps): # Run for 1000 test steps
    action = agent.choose_action(state)
    next_state, reward, _, truncated, _ = env.step(action)
    total_test_reward += reward
    state = next_state
    
    if truncated:
        state, _ = env.reset()
        
print(f"Total reward over {total_test_steps} test steps: {total_test_reward}")
print(f"Average reward per step: {total_test_reward / total_test_steps:.2f}")

# --- 4. Save the Agent ---
print("Saving trained agent...")
agent.save_q_table("ai_traffic_agent.json")
print("Done.")