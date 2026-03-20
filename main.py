from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# 1. Initialize the FastAPI app
app = FastAPI(title="Suryapet Weather Prediction API")

# 2. Add CORS Middleware (Crucial for connecting to Next.js later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your frontend URL (e.g., "http://localhost:3000")
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load the trained model and scaler into memory on startup
print("Loading model and scaler...")
# Ensure your model is named exactly this in the same folder
model = load_model("suryapet_lstm_weather.h5") 

with open("weather_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("Artifacts loaded successfully!")

# 4. Define the expected JSON payload format
class WeatherInput(BaseModel):
    # We expect a 2D list: 14 days, each containing 5 weather features
    data: list[list[float]]

# 5. The Prediction Endpoint
@app.post("/predict")
def predict_weather(payload: WeatherInput):
    # Convert incoming JSON list to a NumPy array
    input_data = np.array(payload.data)
    
    # Validation: Ensure the frontend sent exactly 14 days of 5 features
    if input_data.shape != (14, 5):
        raise HTTPException(status_code=400, detail=f"Expected shape (14, 5), got {input_data.shape}")
        
    # Scale the raw input data using our saved training scaler
    scaled_input = scaler.transform(input_data)
    
    # Reshape into the 3D tensor the LSTM expects: (1 batch, 14 time_steps, 5 features)
    model_input = scaled_input.reshape(1, 14, 5)
    
    # Generate the scaled prediction
    scaled_prediction = model.predict(model_input)
    
    # Inverse-scale the prediction back to actual degrees Celsius
    dummy = np.zeros((1, 5))
    dummy[0, 0] = scaled_prediction[0][0] # Place prediction in the Temperature column
    actual_temp = scaler.inverse_transform(dummy)[0, 0]
    
    # Return the clean, formatted response
    return {
        "location": "Suryapet",
        "predicted_temperature_celsius": round(float(actual_temp), 2),
        "status": "success"
    }