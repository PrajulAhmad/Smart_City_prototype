import time
import json
import random
import paho.mqtt.client as mqtt
import math  # We need this for the time simulation

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
SIMULATED_DAY_LENGTH = 60 # A full day cycle lasts 60 seconds
SENSORS = {
    "SENSOR-001": "smartcity/traffic/sensor1/data",
    "SENSOR-002": "smartcity/traffic/sensor2/data"
}

# --- Main Logic ---
def on_connect(client, userdata, flags, rc):
    print(f"Simulator connected to {MQTT_BROKER}")

def get_realistic_traffic(sensor_id, time_of_day_factor):
    """
    Calculates vehicle count based on time of day (0.0 to 1.0)
    time_of_day_factor will pulse from 0 (night) to 1 (peak rush hour) twice.
    """
    if sensor_id == "SENSOR-001": # Quiet Street
        # Base traffic (night): 10-20
        base_traffic = random.randint(10, 20)
        # Rush hour adds up to 40 extra vehicles
        rush_hour_traffic = int(time_of_day_factor * 40)
        vehicle_count = base_traffic + rush_hour_traffic
        # Total range: 10 (night) to 60 (peak)
        
    else: # Busy Street
        # Base traffic (night): 20-40
        base_traffic = random.randint(20, 40)
        # Rush hour adds up to 60 extra vehicles
        rush_hour_traffic = int(time_of_day_factor * 60)
        vehicle_count = base_traffic + rush_hour_traffic
        # Total range: 20 (night) to 100 (peak)
        
    return vehicle_count

def simulate_sensor_data(sensor_id):
    # Calculate the current "time of day" in our 60-second cycle
    time_of_day_normalized = (time.time() % SIMULATED_DAY_LENGTH) / SIMULATED_DAY_LENGTH
    
    # Convert to a radian (0 to 2*PI)
    radian = time_of_day_normalized * 2 * math.pi
    
    # Use a sine wave to simulate two peaks (morning/evening)
    # This formula goes from 0 -> 1 -> 0 -> 1 -> 0
    traffic_factor = (math.sin(radian * 2 - (math.pi / 2)) + 1) / 2
    
    # Get the vehicle count based on this time
    vehicle_count = get_realistic_traffic(sensor_id, traffic_factor)
    
    data = {
        "sensor_id": sensor_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "vehicle_count": vehicle_count,
        "time_of_day_factor": round(traffic_factor, 2) # For debugging
    }
    return json.dumps(data)

# --- Execution ---
if __name__ == "__main__":
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    print(f"Simulating realistic 24-hour (60-sec) cycles. Press CTRL+C to stop.")
    try:
        while True:
            for sensor_id, topic in SENSORS.items():
                payload = simulate_sensor_data(sensor_id)
                client.publish(topic, payload)
                # We print the payload to see the cycle
                print(f"Published from {sensor_id}: {payload}")
            
            time.sleep(2) # Publish data every 2 seconds

    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        client.loop_stop()
        client.disconnect()