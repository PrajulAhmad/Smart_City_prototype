import time
import json
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import sys
import os
import random 

# --- PASTE THE TrafficEnv CLASS HERE ---
class TrafficEnv():
    def __init__(self):
        self.cars_waiting = 0
        self.light_state = 0 # 0 = Red, 1 = Green
        self.state_space_cars = 6
        self.state_space_light = 2
        self.action_space_n = 2 # 0 = Set to Red, 1 = Set to Green

    def get_state_index(self):
        if self.cars_waiting == 0: car_index = 0
        elif self.cars_waiting <= 50: car_index = 1
        elif self.cars_waiting <= 100: car_index = 2
        elif self.cars_waiting <= 150: car_index = 3
        elif self.cars_waiting <= 200: car_index = 4
        else: car_index = 5
        return (car_index, self.light_state)

    def reset(self):
        self.cars_waiting = 0
        self.light_state = 0
        return self.get_state_index()

    def step(self, action):
        self.light_state = action
        cars_cleared = 0
        reward = -self.cars_waiting 
        if self.light_state == 1: # Green
            cars_cleared = min(self.cars_waiting, 30) 
            self.cars_waiting -= cars_cleared
            reward += (cars_cleared * 20) 
        new_state_index = self.get_state_index()
        return new_state_index, reward
    
    def add_cars(self, num_cars):
        self.cars_waiting = min(255, self.cars_waiting + num_cars)
# --- END OF TrafficEnv CLASS ---

# --- Q-Learning Agent ---
class QAgent():
    def __init__(self, env):
        self.q_table = np.zeros((env.state_space_cars, env.state_space_light, env.action_space_n))
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def choose_action(self, state_index):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 1) # Explore
        else:
            return np.argmax(self.q_table[state_index]) # Exploit

    def learn(self, state_index, action, reward, new_state_index):
        old_value = self.q_table[state_index + (action,)]
        next_max = np.max(self.q_table[new_state_index])
        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state_index + (action,)] = new_value
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, path):
        np.save(path, self.q_table)
        
    def load(self, path):
        self.q_table = np.load(path)
        self.epsilon = self.epsilon_min 
# --- END OF QAgent CLASS ---


# --- Edge Node Configuration ---
if len(sys.argv) != 3:
    print("Usage: python edge_node_fl.py <NODE_ID> <DATA_TOPIC>")
    sys.exit(1)
    
NODE_ID = sys.argv[1]
DATA_TOPIC = sys.argv[2]
# --- End Configuration ---

RL_AGENT_SAVE_PATH = f"rl_agent_q_table_{NODE_ID}.npy" 

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
WEIGHTS_TOPIC = "smartcity/fl/weights"
GLOBAL_MODEL_TOPIC = "smartcity/fl/global_model"
PRED_TOPIC = "smartcity/fl/predictions"
LOSS_TOPIC = "smartcity/fl/metrics"
RL_ACTION_TOPIC = "smartcity/rl/action"

# --- Global Variables ---
LOCAL_BATCH_SIZE = 10
local_data_batch = [] 
local_labels_batch = []
fl_model = None
env = None
rl_agent = None 
rl_state = None 
client = None # Make client global

# --- 1. Define the FL Model ---
def create_fl_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)), # Expects (batch_size, 1)
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- NEW: Helper function for Job 1 (RL) ---
def run_rl_step(vehicle_count):
    global rl_agent, env, rl_state, client
    
    env.add_cars(vehicle_count)
    current_state_index = env.get_state_index()
    action = rl_agent.choose_action(current_state_index)
    new_state_index, reward = env.step(action)
    rl_agent.learn(current_state_index, action, reward, new_state_index)

    action_payload = json.dumps({
        "node_id": NODE_ID,
        "action": int(action),
        "action_str": "GREEN" if int(action) == 1 else "RED",
        "cars_waiting": env.cars_waiting,
        "reward": reward
    })
    client.publish(RL_ACTION_TOPIC, action_payload)

# --- NEW: Helper function for Job 2 (FL) ---
def run_fl_step(vehicle_count):
    global local_data_batch, local_labels_batch, fl_model, client
    
    scaled_count_for_predict = np.array([[vehicle_count / 100.0]])
    scaled_count_for_batch = vehicle_count / 100.0
    label = 1 if vehicle_count > 55 else 0
    
    prediction = fl_model.predict(scaled_count_for_predict, verbose=0)
    pred_payload = json.dumps({ "node_id": NODE_ID, "vehicle_count": vehicle_count, "prediction": float(prediction[0][0]) })
    client.publish(PRED_TOPIC, pred_payload)
    
    local_data_batch.append(scaled_count_for_batch)
    local_labels_batch.append(label)

    if len(local_data_batch) >= LOCAL_BATCH_SIZE:
        print(f"\n--- [{NODE_ID}] FL Batch full. Training local model... ---")
        
        X_train = np.array(local_data_batch).reshape(-1, 1)
        
        # --- THIS IS THE FIX ---
        # We reshape y_train to (10, 1) to match the model's output shape
        y_train = np.array(local_labels_batch).reshape(-1, 1)
        # --- END OF FIX ---
        
        history = fl_model.fit(X_train, y_train, epochs=3, batch_size=2, verbose=0)
        final_loss = history.history['loss'][-1]
        
        loss_payload = json.dumps({ "node_id": NODE_ID, "timestamp": time.strftime("%Y-m-%dT%H:%M:%SZ", time.gmtime()), "loss": final_loss })
        client.publish(LOSS_TOPIC, loss_payload) 
        
        print(f"--- [{NODE_ID}] FL Training complete (Loss: {final_loss:.4f}). Publishing weights... ---")
        weights = fl_model.get_weights()
        payload = pickle.dumps(weights)
        client.publish(f"{WEIGHTS_TOPIC}/{NODE_ID}", payload)
        
        local_data_batch = []
        local_labels_batch = []

# --- 3. MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"[{NODE_ID}] Connected to {MQTT_BROKER}")
    client.subscribe(DATA_TOPIC)
    client.subscribe(GLOBAL_MODEL_TOPIC)
    print(f"[{NODE_ID}] Subscribed to {DATA_TOPIC} and {GLOBAL_MODEL_TOPIC}")

def on_message(client, userdata, msg):
    global fl_model

    try:
        # --- Message 1: Received new GLOBAL MODEL from Cloud ---
        if msg.topic == GLOBAL_MODEL_TOPIC:
            print(f"\n--- [{NODE_ID}] Received new Global Model from Cloud! ---")
            new_weights = pickle.loads(msg.payload)
            fl_model.set_weights(new_weights)
            print(f"--- [{NODE_ID}] FL model updated with new weights. ---\n")
            return

        # --- Message 2: Received new SENSOR DATA ---
        if msg.topic == DATA_TOPIC:
            data = json.loads(msg.payload.decode())
            vehicle_count = data['vehicle_count']
            
            # --- Job 1: Reinforcement Learning ---
            try:
                run_rl_step(vehicle_count)
            except Exception as e:
                print(f"[{NODE_ID}] ERROR IN RL JOB: {e}")
            
            # --- Job 2: Federated Learning ---
            try:
                run_fl_step(vehicle_count)
            except Exception as e:
                print(f"[{NODE_ID}] ERROR IN FL JOB: {e}")

    except Exception as e:
        print(f"[{NODE_ID}] CRITICAL Error in on_message: {e}")

# --- Execution ---
if __name__ == "__main__":
    # --- Initialize FL ---
    fl_model = create_fl_model()
    print(f"[{NODE_ID}] Federated Learning (FL) model created.")
    
    # --- Initialize RL ---
    env = TrafficEnv()
    rl_state = env.reset()
    rl_agent = QAgent(env)
    
    if os.path.exists(RL_AGENT_SAVE_PATH):
        print(f"--- [{NODE_ID}] Found saved Q-Table! Loading... ---")
        rl_agent.load(RL_AGENT_SAVE_PATH)
        print(f"--- [{NODE_ID}] Q-Table reloaded from file. ---")
    else:
        print(f"--- [{NODE_ID}] No saved Q-Table found. Creating new one... ---")
    
    # --- Initialize MQTT ---
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1) # Set client globally
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    
    # --- Graceful Shutdown ---
    try:
        print(f"[{NODE_ID}] System running. Press CTRL+C to stop.")
        client.loop_forever()
    except KeyboardInterrupt:
        print(f"\n--- [{NODE_ID}] SHUTDOWN DETECTED! ---")
        
        print(f"--- [{NODE_ID}] Saving RL agent's Q-Table before exit... ---")
        rl_agent.save(RL_AGENT_SAVE_PATH)
        print(f"--- [{NODE_ID}] Save complete. Goodbye! ---")
        
        client.disconnect()
        sys.exit(0)