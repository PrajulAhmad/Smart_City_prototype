import time
import numpy as np
from WasteEnv import WasteEnv
from RLAgent import QLearningAgent
from MockWastePredictor import MockWastePredictor

# --- Helper Functions (copied from evaluate.py) ---

def train_agent(env, total_steps=100_000, quiet=False):
    """Trains a new Q-Learning agent on a given environment."""
    agent = QLearningAgent(action_space=env.action_space, epsilon_decay=0.9999)
    state, _ = env.reset()
    start_time = time.time()
    for step in range(total_steps):
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        agent.update_epsilon()
        if truncated: state, _ = env.reset()
        if not quiet and (step + 1) % 50_000 == 0: # Updated log interval
            print(f"... training step {step+1}/{total_steps}")
    end_time = time.time()
    if not quiet:
        print(f"Training finished in {end_time - start_time:.2f} seconds.")
    return agent

def test_agent(agent, env, total_steps=5000):
    """Tests a trained agent on an environment."""
    agent.epsilon = 0.0 # Exploitation mode
    state, _ = env.reset()
    total_reward = 0
    for _ in range(total_steps):
        action = agent.choose_action(state)
        next_state, reward, _, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state
        if truncated: state, _ = env.reset()
    return total_reward / total_steps # Return average reward per step

# --- Main Evaluation Script ---

if __name__ == "__main__":
    
    NUM_BINS = 5
    # UPDATED training steps
    TRAINING_STEPS = 500_000 
    TEST_STEPS = 10_000
    
    print("--- 1. Evaluating Baseline (Reactive) Waste Agent ---")
    baseline_env = WasteEnv(num_bins=NUM_BINS, predictor=None)
    print(f"Training Baseline Agent for {TRAINING_STEPS} steps...")
    baseline_agent = train_agent(baseline_env, total_steps=TRAINING_STEPS)
    
    # 1c. Save Baseline Agent (CRITICAL for dashboard)
    baseline_agent.save_q_table("baseline_waste_agent.json")
    
    print(f"Testing Baseline Agent for {TEST_STEPS} steps...")
    baseline_reward = test_agent(baseline_agent, baseline_env, total_steps=TEST_STEPS)
    
    print("\n--- 2. Evaluating AI-Powered (Proactive) Waste Agent ---")
    ai_predictor = MockWastePredictor(num_bins=NUM_BINS)
    ai_env = WasteEnv(num_bins=NUM_BINS, predictor=ai_predictor)
    print(f"Training AI-Powered Agent for {TRAINING_STEPS} steps...")
    ai_agent = train_agent(ai_env, total_steps=TRAINING_STEPS)
    
    print(f"Testing AI-Powered Agent for {TEST_STEPS} steps...")
    ai_reward = test_agent(ai_agent, ai_env, total_steps=TEST_STEPS)
    
    # --- 3. Final Results ---
    print("\n--- 📊 Waste Management Evaluation Complete ---")
    print("Comparing average reward per step (higher is better)\n")
    
    print("=" * 40)
    print(f"| {'Model':<20} | {'Avg. Reward':<15} |")
    print("-" * 40)
    print(f"| {'Baseline (Reactive)':<20} | {baseline_reward:<15.3f} |")
    print(f"| {'AI-Powered (Proactive)':<20} | {ai_reward:<15.3f} |")
    print("=" * 40)
    
    improvement = ((ai_reward - baseline_reward) / abs(baseline_reward)) * 100
    print(f"\nResult: The AI-Powered agent performed {improvement:.2f}% better.")