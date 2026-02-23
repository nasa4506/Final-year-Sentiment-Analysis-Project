from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
from pydantic import BaseModel
from backend.src.models.fused_model import predict_fused_sentiment
from PIL import Image
import io
import logging

router = APIRouter(prefix="/analyze", tags=["Fused Analysis"])
logger = logging.getLogger(__name__)

class FusedResponse(BaseModel):
    sentiment: str
    confidence: float
    details: Optional[dict] = None # To return individual model results if needed
    reasoning: list = []
    math_breakdown: list = []

@router.post("/fused", response_model=FusedResponse)
async def analyze_fused(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    if not any([text, audio_file, image_file]):
        raise HTTPException(status_code=400, detail="At least one input (text, audio, or image) must be provided")
    
    try:
        # Process inputs
        audio_bytes = None
        if audio_file:
            audio_bytes = await audio_file.read()
            
        image = None
        if image_file:
            image_bytes = await image_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
        # Predict
        sentiment, confidence, reasoning, math_breakdown = predict_fused_sentiment(
            text=text,
            audio_bytes=audio_bytes,
            image=image
        )
        
        return FusedResponse(sentiment=sentiment, confidence=confidence, reasoning=reasoning, math_breakdown=math_breakdown)
    except Exception as e:
        logger.error(f"Error analyzing fused input: {e}")
        raise HTTPException(status_code=500, detail=str(e))
