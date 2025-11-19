**🏙️ AI-Powered Smart City Management System**

A lightweight and practical prototype that uses prediction + reinforcement learning to manage key city operations such as traffic, waste, and energy.
The idea is simple: let edge devices make smarter decisions by combining local sensor data with AI-generated predictions.

🚀 What This Project Does

Simulates a smart city with custom Gym environments
Uses a mock Federated Learning model to generate predictions
Uses Q-Learning for decision-making
Shows everything on a real-time dashboard
Compares Baseline vs AI-Powered agents

🧠 How It Works

The system has four main parts:
IoT Simulation – Traffic, Waste, and Energy environments
Prediction Layer – Mock FL models that forecast upcoming demand
Reinforcement Learning – Agents choose the best action based on state + prediction
Dashboard – A Flask UI showing real-time performance

🔮 Future Work

Upgrade Q-Learning → Deep Q-Network (DQN)
Replace mock predictors with real Federated Learning
Enable multi-agent interactions across domains

**🚀 How to Use**

1) Clone the repository

2) Install dependencies
pip install -r requirements.txt

Step 1: Train the agents
python3 train.py
python3 evaluate.py

python3 train_waste.py
python3 evaluate_waste.py

python3 train_energy.py
python3 evaluate_energy.py

This generates the Q-tables (saved as .json files).

Step 2: Start the dashboard
python3 app.py

Step 3: Open the visualization
Go to:
http://127.0.0.1:5000




📁 Key Files

RLAgent.py          # Q-Learning agent

TrafficEnv.py       # Traffic simulation

WasteEnv.py         # Waste simulation

EnergyEnv.py        # Energy simulation

app.py              # Dashboard server

train*.py           # Training scripts

evaluate*.py        # Baseline comparison
