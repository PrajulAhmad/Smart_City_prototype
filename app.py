from flask import Flask, render_template, jsonify
import numpy as np

# Import all our project classes
from RLAgent import QLearningAgent
from TrafficEnv import TrafficIntersectionEnv
from MockPredictor import MockTrafficPredictor
from WasteEnv import WasteEnv
from MockWastePredictor import MockWastePredictor
from EnergyEnv import EnergyEnv
from MockEnergyPredictor import MockEnergyPredictor

# --- 1. Initialize our Flask App ---
app = Flask(__name__)

# --- 2. Load ALL Models and Create Environments ---
print("Loading all 6 agents and 3 environments...")

# --- Traffic ---
ai_traffic_predictor = MockTrafficPredictor()
ai_traffic_env = TrafficIntersectionEnv(predictor=ai_traffic_predictor)
ai_traffic_agent = QLearningAgent(action_space=ai_traffic_env.action_space)
ai_traffic_agent.load_q_table("ai_traffic_agent.json")
ai_traffic_state, _ = ai_traffic_env.reset()
ai_traffic_reward = 0

baseline_traffic_env = TrafficIntersectionEnv(predictor=None)
baseline_traffic_agent = QLearningAgent(action_space=baseline_traffic_env.action_space)
baseline_traffic_agent.load_q_table("baseline_traffic_agent.json")
baseline_traffic_state, _ = baseline_traffic_env.reset()
baseline_traffic_reward = 0

# --- Waste ---
NUM_BINS = 5
ai_waste_predictor = MockWastePredictor(num_bins=NUM_BINS)
ai_waste_env = WasteEnv(num_bins=NUM_BINS, predictor=ai_waste_predictor)
ai_waste_agent = QLearningAgent(action_space=ai_waste_env.action_space)
ai_waste_agent.load_q_table("ai_waste_agent.json")
ai_waste_state, _ = ai_waste_env.reset()
ai_waste_reward = 0

baseline_waste_env = WasteEnv(num_bins=NUM_BINS, predictor=None)
baseline_waste_agent = QLearningAgent(action_space=baseline_waste_env.action_space)
baseline_waste_agent.load_q_table("baseline_waste_agent.json")
baseline_waste_state, _ = baseline_waste_env.reset()
baseline_waste_reward = 0

# --- Energy ---
ai_energy_predictor = MockEnergyPredictor()
ai_energy_env = EnergyEnv(predictor=ai_energy_predictor)
ai_energy_agent = QLearningAgent(action_space=ai_energy_env.action_space)
ai_energy_agent.load_q_table("ai_energy_agent.json")
ai_energy_state, _ = ai_energy_env.reset()
ai_energy_reward = 0

baseline_energy_env = EnergyEnv(predictor=None)
baseline_energy_agent = QLearningAgent(action_space=baseline_energy_env.action_space)
baseline_energy_agent.load_q_table("baseline_energy_agent.json")
baseline_energy_state, _ = baseline_energy_env.reset()
baseline_energy_reward = 0

print("Server is ready. Navigate to http://127.0.0.1:5000")

# --- 3. Define Webpage Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/step')
def simulation_step():
    """
    Runs one step of ALL SIX simulations and returns data.
    """
    global ai_traffic_state, ai_traffic_reward, baseline_traffic_state, baseline_traffic_reward
    global ai_waste_state, ai_waste_reward, baseline_waste_state, baseline_waste_reward
    global ai_energy_state, ai_energy_reward, baseline_energy_state, baseline_energy_reward

    # --- Run Traffic ---
    ai_action = ai_traffic_agent.choose_action(ai_traffic_state)
    ai_traffic_state, ai_step_reward, _, trunc, _ = ai_traffic_env.step(ai_action)
    ai_traffic_reward += ai_step_reward
    if trunc: ai_traffic_state, _ = ai_traffic_env.reset()
    
    base_action = baseline_traffic_agent.choose_action(baseline_traffic_state)
    baseline_traffic_state, base_step_reward, _, trunc, _ = baseline_traffic_env.step(base_action)
    baseline_traffic_reward += base_step_reward
    if trunc: baseline_traffic_state, _ = baseline_traffic_env.reset()
    
    traffic_data = {
        "baseline": {
            "reward": f"{baseline_traffic_reward:.2f}", 
            "step_reward": float(base_step_reward), # <-- FIX
            "phase": "N/S Green" if baseline_traffic_state[2] == 0 else "E/W Green", 
            "cars_ns": int(baseline_traffic_state[0]),
            "cars_ew": int(baseline_traffic_state[1])
        },
        "ai": {
            "reward": f"{ai_traffic_reward:.2f}", 
            "step_reward": float(ai_step_reward), # <-- FIX
            "phase": "N/S Green" if ai_traffic_state[2] == 0 else "E/W Green", 
            "cars_ns": int(ai_traffic_state[0]),
            "cars_ew": int(ai_traffic_state[1]),
            "pred_ns": int(ai_traffic_state[4]),
            "pred_ew": int(ai_traffic_state[5])
        }
    }

    # --- Run Waste ---
    ai_action = ai_waste_agent.choose_action(ai_waste_state)
    ai_waste_state, ai_step_reward, _, trunc, _ = ai_waste_env.step(ai_action)
    ai_waste_reward += ai_step_reward
    if trunc: ai_waste_state, _ = ai_waste_env.reset()
    
    base_action = baseline_waste_agent.choose_action(baseline_waste_state)
    baseline_waste_state, base_step_reward, _, trunc, _ = baseline_waste_env.step(base_action)
    baseline_waste_reward += base_step_reward
    if trunc: baseline_waste_state, _ = baseline_waste_env.reset()
    
    waste_data = {
        "baseline": {
            "reward": f"{baseline_waste_reward:.2f}", 
            "step_reward": float(base_step_reward), # <-- FIX
            "truck_at": int(baseline_waste_state[0]),
            "bins": str(baseline_waste_state[1:])
        },
        "ai": {
            "reward": f"{ai_waste_reward:.2f}", 
            "step_reward": float(ai_step_reward), # <-- FIX
            "truck_at": int(ai_waste_state[0]),
            "bins": str(ai_waste_state[1:NUM_BINS+1]), 
            "preds": str(ai_waste_state[NUM_BINS+1:])
        }
    }

    # --- Run Energy ---
    ai_action = ai_energy_agent.choose_action(ai_energy_state)
    ai_energy_state, ai_step_reward, _, trunc, _ = ai_energy_env.step(ai_action)
    ai_energy_reward += ai_step_reward
    if trunc: ai_energy_state, _ = ai_energy_env.reset()
    
    base_action = baseline_energy_agent.choose_action(baseline_energy_state)
    baseline_energy_state, base_step_reward, _, trunc, _ = baseline_energy_env.step(base_action)
    baseline_energy_reward += base_step_reward
    if trunc: baseline_energy_state, _ = baseline_energy_env.reset()
    
    energy_data = {
        "baseline": {
            "reward": f"{baseline_energy_reward:.2f}", 
            "step_reward": float(base_step_reward), # <-- FIX
            "hour": int(baseline_energy_state[0]),
            "battery": int(baseline_energy_state[1])
        },
        "ai": {
            "reward": f"{ai_energy_reward:.2f}", 
            "step_reward": float(ai_step_reward), # <-- FIX
            "hour": int(ai_energy_state[0]),
            "battery": int(ai_energy_state[1]),
            "pred_demand": int(ai_energy_state[2])
        }
    }
    
    # --- Return all data ---
    return jsonify({
        "traffic": traffic_data,
        "waste": waste_data,
        "energy": energy_data
    })

# --- 4. Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
