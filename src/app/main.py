"""
Unplug API - Smartphone Addiction Prediction API
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path
from src.model.model import UnplugPredictor
import uvicorn
import sys

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))


app = FastAPI(
    title="Unplug",
    description="Не пора ли отложить телефон?",
    version="1.0.0"
)

predictor = UnplugPredictor()

STATIC_DIR = BASE_DIR / "static"

class PredictionRequest(BaseModel):
    """
    Model for requests.
    """
    daily_screen_time: float = Field(..., ge=0, le=24)
    weekend_screen_time: float = Field(..., ge=0, le=24)
    social_media_hours: float = Field(..., ge=0, le=24)

class PredictionResponse(BaseModel):
    """
    Model for responses.
    """
    addicted: bool
    message: str
    result: str
    probability: float

@app.get("/")
async def root():
    """Returns the main application page."""
    return FileResponse(STATIC_DIR / "index.html")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    A function that uses model to predict addition.

    Args:
        request: PredictionRequest
    Returns:
        response: PredictionResponse
    """
    try:
        result = predictor.predict(
            request.daily_screen_time,
            request.weekend_screen_time,
            request.social_media_hours
        )

        return PredictionResponse(
            addicted=result['addicted'],
            message=result['message'],
            result=result['result'],
            probability=result['probability']
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Health-check"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)