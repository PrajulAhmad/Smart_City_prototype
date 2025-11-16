import json
from jinja2 import Environment, FileSystemLoader # We used this to make the first HTML

# Import our classes
from TrafficEnv import TrafficIntersectionEnv
from RLAgent import QLearningAgent
from MockPredictor import MockTrafficPredictor

print("Generating dashboard report...")

# --- 1. Load Trained Agent ---
predictor = MockTrafficPredictor()
env = TrafficIntersectionEnv(predictor=predictor)
agent = QLearningAgent(action_space=env.action_space)
agent.load_q_table("ai_traffic_agent.json") # Load our trained brain

# --- 2. Run Simulation and Log Data ---
simulation_data = []
state, _ = env.reset()

for step in range(100): # Generate a 100-step report
    action = agent.choose_action(state)
    next_state, reward, _, truncated, _ = env.step(action)

    # Log everything
    log_entry = {
        "step": step,
        "cars_ns": state[0],
        "cars_ew": state[1],
        "phase": "N/S Green" if state[2] == 0 else "E/W Green",
        "time_in_phase": state[3],
        "predicted_ns": state[4], # <-- NEW DATA
        "predicted_ew": state[5], # <-- NEW DATA
        "action_taken": "Stay" if action == 0 else "Switch",
        "reward": f"{reward:.2f}"
    }
    simulation_data.append(log_entry)

    state = next_state
    if truncated:
        state, _ = env.reset()

# --- 3. Generate HTML from Template ---
# This assumes your template is named "Dashboard_fl_template.html"
env_jinja = Environment(loader=FileSystemLoader('.'))
template = env_jinja.get_template("Dashboard_fl_template.html")

html_output = template.render(
    report_title="AI-Powered Traffic Control Report",
    simulation_log=simulation_data
)

# Save the final HTML file
with open("Dashboard_fl.html", "w") as f:
    f.write(html_output)

print(f"Successfully generated Dashboard_fl.html with {len(simulation_data)} steps.")
