from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import pickle

# --- THE FIX: Create a custom Dense layer to ignore the new Keras parameter ---
class SafeDense(Dense):
    def __init__(self, **kwargs):
        # Remove 'quantization_config' before passing it to the older Keras version
        kwargs.pop('quantization_config', None)
        super().__init__(**kwargs)

# 1. Initialize the FastAPI app
app = FastAPI(title="Suryapet Weather Prediction API")

# 2. Add CORS Middleware 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the trained model and scaler into memory on startup
print("Loading model and scaler...")

# --- THE FIX: Tell Keras to use our SafeDense class when it builds the model ---
model = load_model("suryapet_lstm_weather.h5", custom_objects={'Dense': SafeDense}) 

with open("weather_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Artifacts loaded successfully!")

# 4. Define the expected JSON payload format
class WeatherInput(BaseModel):
    data: list[list[float]]

# 5. The Prediction Endpoint
@app.post("/predict")
def predict_weather(payload: WeatherInput):
    input_data = np.array(payload.data)
    
    if input_data.shape != (14, 5):
        raise HTTPException(status_code=400, detail=f"Expected shape (14, 5), got {input_data.shape}")
        
    scaled_input = scaler.transform(input_data)
    model_input = scaled_input.reshape(1, 14, 5)
    
    scaled_prediction = model.predict(model_input)
    
    dummy = np.zeros((1, 5))
    dummy[0, 0] = scaled_prediction[0][0] 
    actual_temp = scaler.inverse_transform(dummy)[0, 0]
    
    return {
        "location": "Suryapet",
        "predicted_temperature_celsius": round(float(actual_temp), 2),
        "status": "success"
    }