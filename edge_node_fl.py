import time
import json
import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import sys

# --- Edge Node Configuration ---
if len(sys.argv) != 3:
    print("Usage: python edge_node_fl.py <NODE_ID> <DATA_TOPIC>")
    sys.exit(1)
    
NODE_ID = sys.argv[1]
DATA_TOPIC = sys.argv[2]
# --- End Configuration ---

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
WEIGHTS_TOPIC = "smartcity/fl/weights"
GLOBAL_MODEL_TOPIC = "smartcity/fl/global_model"
# --- NEW TOPICS FOR VISUALIZATION ---
PRED_TOPIC = "smartcity/fl/predictions"
LOSS_TOPIC = "smartcity/fl/metrics"
# ---

# --- Federated Learning Configuration ---
LOCAL_BATCH_SIZE = 10
local_data_batch = []
local_labels_batch = []

# --- 1. Define the ML Model ---
def create_model():
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
print(f"[{NODE_ID}] Local model created.")

# --- 2. MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"[{NODE_ID}] Connected to {MQTT_BROKER}")
    client.subscribe(DATA_TOPIC)
    client.subscribe(GLOBAL_MODEL_TOPIC)
    print(f"[{NODE_ID}] Subscribed to {DATA_TOPIC} and {GLOBAL_MODEL_TOPIC}")

def on_message(client, userdata, msg):
    global local_data_batch, local_labels_batch, model

    try:
        # --- Message 1: Received new GLOBAL MODEL from Cloud ---
        if msg.topic == GLOBAL_MODEL_TOPIC:
            print(f"\n--- [{NODE_ID}] Received new Global Model from Cloud! ---")
            new_weights = pickle.loads(msg.payload)
            model.set_weights(new_weights)
            print(f"--- [{NODE_ID}] Local model updated with new weights. ---\n")
            return

        # --- Message 2: Received new SENSOR DATA ---
        if msg.topic == DATA_TOPIC:
            data = json.loads(msg.payload.decode())
            vehicle_count = data['vehicle_count']
            scaled_count = np.array([vehicle_count / 100.0])
            label = 1 if vehicle_count > 55 else 0
            
            # --- NEW: Publish real-time prediction ---
            prediction = model.predict(scaled_count, verbose=0)
            pred_payload = json.dumps({
                "node_id": NODE_ID,
                "vehicle_count": vehicle_count,
                "prediction": float(prediction[0][0])
            })
            client.publish(PRED_TOPIC, pred_payload)
            # ---
            
            print(f"[{NODE_ID}] Data: {vehicle_count} -> Pred: {prediction[0][0]:.2f} (Label: {label})")

            local_data_batch.append(scaled_count)
            local_labels_batch.append(label)

            # --- 3. Local Training & Weight Publishing ---
            if len(local_data_batch) >= LOCAL_BATCH_SIZE:
                print(f"\n--- [{NODE_ID}] Batch full. Training local model... ---")
                X = np.array(local_data_batch)
                y = np.array(local_labels_batch)
                
                # --- NEW: Capture training history ---
                history = model.fit(X, y, epochs=3, batch_size=2, verbose=0)
                final_loss = history.history['loss'][-1]
                
                loss_payload = json.dumps({
                    "node_id": NODE_ID,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "loss": final_loss
                })
                client.publish(LOSS_TOPIC, loss_payload)
                # ---
                
                print(f"--- [{NODE_ID}] Training complete (Loss: {final_loss:.4f}). Publishing weights... ---")

                weights = model.get_weights()
                payload = pickle.dumps(weights)
                client.publish(f"{WEIGHTS_TOPIC}/{NODE_ID}", payload)
                
                local_data_batch = []
                local_labels_batch = []

    except Exception as e:
        print(f"[{NODE_ID}] Error processing message: {e}")

# --- Execution ---
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
