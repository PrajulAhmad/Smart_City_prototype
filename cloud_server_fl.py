import json
import time
import paho.mqtt.client as mqtt
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
WEIGHTS_TOPIC = "smartcity/fl/weights/#"          # Listen for all weights
GLOBAL_MODEL_TOPIC = "smartcity/fl/global_model"  # Publish global model here
SYSTEM_EVENTS_TOPIC = "smartcity/fl/system_events"  # Publish events here

# --- Federated Learning Configuration ---
MIN_NODES_FOR_AGGREGATION = 2  # Wait for 2 nodes before averaging
weight_updates_cache = {}       # Use a dict to store weights by node_id


# --- 1. Define the Global Model ---
def create_model():
    """Defines the Keras model. MUST be identical to the edge node model."""
    model = keras.Sequential([
        layers.Dense(16, activation='relu', input_shape=(1,)),
        layers.Dense(8, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


global_model = create_model()
print("[CLOUD] Global model created and waiting for weights.")


# --- 2. Federated Aggregation Logic ---
def federated_average(weight_updates_list):
    """
    Averages the weights from all edge nodes.
    This is the core of Federated Averaging (FedAvg).
    """
    print(f"\n--- [CLOUD] Aggregating weights from {len(weight_updates_list)} nodes... ---")

    num_layers = len(weight_updates_list[0])
    new_weights = []

    # Iterate over each layer
    for layer_index in range(num_layers):
        # Collect all weights for this specific layer from all nodes
        layer_weights = np.array([node_weights[layer_index] for node_weights in weight_updates_list])

        # Calculate the average for this layer
        avg_layer_weights = np.mean(layer_weights, axis=0)
        new_weights.append(avg_layer_weights)

    print("--- [CLOUD] Weight aggregation complete. ---")
    return new_weights


# --- 3. MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"[CLOUD] Connected to {MQTT_BROKER}")
    client.subscribe(WEIGHTS_TOPIC)
    print(f"[CLOUD] Subscribed to {WEIGHTS_TOPIC}")


def on_message(client, userdata, msg):
    """Handles incoming model weights from edge nodes."""
    global weight_updates_cache, global_model

    try:
        # Get the node_id from the topic
        topic_parts = msg.topic.split('/')
        node_id = topic_parts[-1]

        # Deserialize the weights
        weights = pickle.loads(msg.payload)
        weight_updates_cache[node_id] = weights
        print(f"[CLOUD] Received weights from {node_id}. Cache size: {len(weight_updates_cache)}")

        # --- 4. Aggregation & Publishing ---
        if len(weight_updates_cache) >= MIN_NODES_FOR_AGGREGATION:
            print(f"--- [CLOUD] Minimum nodes reached. Starting aggregation... ---")

            # Perform the federated average
            new_global_weights = federated_average(list(weight_updates_cache.values()))

            # Update the server's global model
            global_model.set_weights(new_global_weights)

            # Serialize the new global model
            payload = pickle.dumps(new_global_weights)

            # Publish the new global model back to all nodes
            client.publish(GLOBAL_MODEL_TOPIC, payload)
            print(f"--- [CLOUD] Published new global model to all nodes! ---")

            # --- Publish system event ---
            event_payload = json.dumps({
                "event": "GLOBAL_MODEL_PUBLISHED",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
            client.publish(SYSTEM_EVENTS_TOPIC, event_payload)

            # Clear the cache for the next round
            weight_updates_cache = {}

    except Exception as e:
        print(f"[CLOUD] Error processing weights: {e}")


# --- 5. Main Execution ---
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
