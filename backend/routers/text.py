from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.src.models.text_model import predict_text_sentiment
import logging

router = APIRouter(prefix="/analyze", tags=["Text Analysis"])
logger = logging.getLogger(__name__)

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    sentiment: str
    confidence: float
    reasoning: list = []

@router.post("/text", response_model=TextResponse)
async def analyze_text(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        sentiment, confidence, reasoning = predict_text_sentiment(request.text)
        return TextResponse(sentiment=sentiment, confidence=confidence, reasoning=reasoning)
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
