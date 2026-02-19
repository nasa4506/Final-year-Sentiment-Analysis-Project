from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from backend.src.models.vision_model import predict_vision_sentiment
from PIL import Image
import io
import logging

router = APIRouter(prefix="/analyze", tags=["Vision Analysis"])
logger = logging.getLogger(__name__)

class VisionResponse(BaseModel):
    sentiment: str
    confidence: float

@router.post("/vision", response_model=VisionResponse)
async def analyze_vision(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image file")
    
    try:
        # Read file bytes and convert to PIL Image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Predict
        sentiment, confidence = predict_vision_sentiment(image)
        
        return VisionResponse(sentiment=sentiment, confidence=confidence)
    except Exception as e:
        logger.error(f"Error analyzing vision: {e}")
        raise HTTPException(status_code=500, detail=str(e))
