import json
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.linear_model import LogisticRegression

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "smartcity/traffic/data"  # Topic to listen to
MQTT_TOPIC_PUB = "smartcity/traffic/cloud_ingest" # Topic to forward data to
EDGE_NODE_ID = "EDGE-001"

# --- Machine Learning Model ---
def train_model():
    """
    Trains a simple Logistic Regression model.
    This simulates loading a pre-trained model on the edge device.
    """
    print("Training a simple ML model for congestion detection...")
    # Mock training data: [vehicle_count]
    X_train = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
    # Mock labels: 0 = Low, 1 = High (Threshold at 55)
    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

# --- MQTT Callbacks ---

def on_connect(client, userdata, flags, rc):
    """Callback for when the client connects."""
    if rc == 0:
        print(f"Edge Node '{EDGE_NODE_ID}' connected to {MQTT_BROKER}")
        # Subscribe to the sensor data topic
        client.subscribe(MQTT_TOPIC_SUB)
        print(f"Subscribed to topic: {MQTT_TOPIC_SUB}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    """Callback for when a message is received."""
    try:
        # 1. Receive and parse data
        payload = msg.payload.decode()
        data = json.loads(payload)
        vehicle_count = data.get("vehicle_count", 0)
        
        # 2. Process at Edge (Apply ML Model)
        # We need to reshape the single data point for the model
        vehicle_count_array = np.array([vehicle_count]).reshape(-1, 1)
        
        # Make a prediction
        prediction_prob = model.predict_proba(vehicle_count_array)[0][1] # Prob of 'High'
        congestion_label = "High" if prediction_prob > 0.5 else "Low"
        
        print(f"\n--- [EDGE] Data Received from {data['sensor_id']} ---")
        print(f"Vehicle Count: {vehicle_count} -> Prediction: {congestion_label} (Prob: {prediction_prob:.2f})")

        # 3. Immediate Action (as per Figure 5)
        if congestion_label == "High":
            print(f"[EDGE_ACTION] Congestion HIGH. Triggering adaptive traffic light signal...")
        else:
            print(f"[EDGE_ACTION] Congestion LOW. Maintaining normal signal timing.")

        # 4. Forward to Cloud for long-term analysis
        # (This is the 'No' path for 'Latency Critical?' or just forwarding)
        client.publish(MQTT_TOPIC_PUB, payload)
        print(f"Forwarded data to cloud topic: {MQTT_TOPIC_PUB}")

    except json.JSONDecodeError:
        print(f"Received malformed JSON: {msg.payload.decode()}")
    except Exception as e:
        print(f"Error processing message: {e}")

# --- Execution ---
if __name__ == "__main__":
    # Train/load the model first
    model = train_model()

    # Set up the MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect to the broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")
        exit()

    print(f"Starting Edge Node '{EDGE_NODE_ID}'. Listening for data...")
    
    # Start the loop to listen for messages
    # This is a blocking call that will run forever
    client.loop_forever()