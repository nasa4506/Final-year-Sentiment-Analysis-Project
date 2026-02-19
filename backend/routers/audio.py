from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.src.models.audio_model import predict_audio_sentiment
import logging

router = APIRouter(prefix="/analyze", tags=["Audio Analysis"])
logger = logging.getLogger(__name__)

class AudioResponse(BaseModel):
    sentiment: str
    confidence: float

@router.post("/audio", response_model=AudioResponse)
async def analyze_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read file bytes
        audio_bytes = await file.read()
        
        # Predict
        sentiment, confidence = predict_audio_sentiment(audio_bytes)
        
        return AudioResponse(sentiment=sentiment, confidence=confidence)
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
