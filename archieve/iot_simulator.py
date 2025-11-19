import time
import json
import random
import paho.mqtt.client as mqtt

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "smartcity/traffic/data"
SENSOR_ID = "TCS-001"

# --- Main Logic ---

def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the broker."""
    if rc == 0:
        print(f"Connected to MQTT Broker at {MQTT_BROKER}")
    else:
        print(f"Failed to connect, return code {rc}")

def simulate_sensor_data():
    """Generates a piece of mock traffic data."""
    vehicle_count = random.randint(20, 100)  # Simulate 20 to 100 vehicles
    data = {
        "sensor_id": SENSOR_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vehicle_count": vehicle_count
    }
    return json.dumps(data)  # Convert the dictionary to a JSON string

# --- Execution ---
if __name__ == "__main__":
    # Set up the MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    
    # Connect to the broker
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")
        exit()

    client.loop_start()  # Start the client's network loop in a background thread

    print(f"Simulating IoT Sensor '{SENSOR_ID}'. Press CTRL+C to stop.")

    try:
        while True:
            # Generate and publish data
            payload = simulate_sensor_data()
            result = client.publish(MQTT_TOPIC, payload)
            
            # Check if publish was successful
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Published: {payload}")
            else:
                print(f"Failed to publish, return code {result.rc}")

            time.sleep(3)  # Wait 3 seconds before sending the next reading

    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        client.loop_stop()
        client.disconnect()
        print("Disconnected from MQTT broker.")