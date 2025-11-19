import uvicorn
import json
import paho.mqtt.client as mqtt
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import List
import pydantic

# --- Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SUB = "smartcity/traffic/cloud_ingest"  # Topic to listen to

DATABASE_URL = "sqlite:///./traffic.db"  # Local SQLite database file

# --- 1. Database Setup (SQLAlchemy) ---
# This section creates the database and table
Base = declarative_base()
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class TrafficData(Base):
    """Database model for our traffic data."""
    __tablename__ = "traffic_logs"
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String, index=True)
    timestamp = Column(DateTime)
    vehicle_count = Column(Integer)

# Create the database tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency to get a DB session for each API request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 2. Pydantic Models (Data Validation) ---
# These models define the shape of the data for the API
class TrafficDataSchema(pydantic.BaseModel):
    sensor_id: str
    timestamp: datetime
    vehicle_count: int

    class Config:
        orm_mode = True # Allows mapping from SQLAlchemy model

# --- 3. FastAPI App ---
app = FastAPI(title="Smart City Cloud API")
# ... after this line:
app = FastAPI(title="Smart City Cloud API")

# ADD THIS ENTIRE BLOCK:
origins = [
    "*"  # This allows all origins, which is fine for local testing.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ... before this line:
# mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)

# --- 4. MQTT Client Logic ---
# This runs in the background to collect data
def on_mqtt_connect(client, userdata, flags, rc):
    """Callback for when the MQTT client connects."""
    if rc == 0:
        print(f"Cloud Server connected to MQTT Broker at {MQTT_BROKER}")
        client.subscribe(MQTT_TOPIC_SUB)
        print(f"Subscribed to cloud topic: {MQTT_TOPIC_SUB}")
    else:
        print(f"Failed to connect to MQTT, return code {rc}")

def on_mqtt_message(client, userdata, msg):
    """Callback for when a message is received from the edge node."""
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        print(f"[CLOUD_INGEST] Received data: {payload}")

        # --- Save to Database ---
        # We create a new DB session just for this message
        db = SessionLocal()
        try:
            db_record = TrafficData(
                sensor_id=data['sensor_id'],
                timestamp=datetime.fromisoformat(data['timestamp'].replace("Z", "+00:00")),
                vehicle_count=data['vehicle_count']
            )
            db.add(db_record)
            db.commit()
            print("Successfully saved data to database.")
        except Exception as e:
            print(f"Error saving to database: {e}")
            db.rollback()
        finally:
            db.close()
        # --- End Save to DB ---

    except json.JSONDecodeError:
        print(f"Received malformed JSON: {msg.payload.decode()}")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")

@app.on_event("startup")
def startup_event():
    """This function runs when the FastAPI server starts."""
    mqtt_client.on_connect = on_mqtt_connect
    mqtt_client.on_message = on_mqtt_message
    
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()  # Starts the MQTT client in a background thread
    except Exception as e:
        print(f"Could not connect to MQTT broker: {e}")

# --- 5. API Endpoints ---
# These are the HTTP URLs you can access with your browser

@app.get("/")
def read_root():
    """Root endpoint to check if the server is running."""
    return {"message": "AI-Powered Smart City Cloud API is running."}

@app.get("/analytics/traffic", response_model=List[TrafficDataSchema])
def get_traffic_data(db: Session = Depends(get_db), limit: int = 100):
    """
    API endpoint to get the latest traffic data.
    This is what the dashboard will call.
    """
    print(f"API request received for /analytics/traffic")
    # Query the database for the last 'limit' records
    data = db.query(TrafficData).order_by(TrafficData.timestamp.desc()).limit(limit).all()
    return data

# --- 6. Run the Server ---
if __name__ == "__main__":
    print("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)