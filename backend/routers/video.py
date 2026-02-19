from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from backend.src.models.text_model import predict_text_sentiment
from backend.src.models.audio_model import predict_audio_sentiment
from backend.src.models.vision_model import predict_vision_sentiment
from backend.src.utils.preprocessing import (
    extract_frames_from_video,
    extract_audio_from_video,
    transcribe_audio
)
from backend.src.config.settings import SENTIMENT_UNIFICATION

router = APIRouter(prefix="/analyze", tags=["Video Max Fusion"])
logger = logging.getLogger(__name__)

class ModalityResult(BaseModel):
    sentiment: str
    confidence: float
    details: Optional[str] = None

class VideoResponse(BaseModel):
    video_sentiment: str
    video_confidence: float
    modalities: Dict[str, ModalityResult]
    frames_analyzed: int
    transcription: str

@router.post("/video", response_model=VideoResponse)
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video file")
    
    try:
        # Read video bytes
        video_bytes = await file.read()
        
        results = {}
        
        # 1. Audio and Text Analysis
        logger.info("Extracting audio from video...")
        audio_bytes = extract_audio_from_video(video_bytes)
        
        if audio_bytes:
            # Audio Sentiment
            audio_sentiment, audio_conf = predict_audio_sentiment(audio_bytes)
            results["audio"] = ModalityResult(sentiment=audio_sentiment, confidence=audio_conf)
            
            # Text Sentiment (via Transcription)
            transcription = transcribe_audio(audio_bytes)
            if transcription and transcription.strip():
                text_sentiment, text_conf = predict_text_sentiment(transcription)
                results["text"] = ModalityResult(
                    sentiment=text_sentiment, 
                    confidence=text_conf,
                    details=transcription
                )
            else:
                transcription = ""
        else:
            transcription = ""
            
        # 2. Vision Analysis (Frame-based)
        logger.info("Extracting frames from video...")
        frames = extract_frames_from_video(video_bytes, max_frames=5)
        
        vision_results = []
        if frames:
            for frame in frames:
                s, c = predict_vision_sentiment(frame)
                vision_results.append((s, c))
            
            # Aggregate Vision Results (Weighted Vote)
            sentiment_counts = {}
            for s, c in vision_results:
                if s not in sentiment_counts:
                    sentiment_counts[s] = 0.0
                sentiment_counts[s] += c
            
            # Best vision sentiment is the one with highest accumulated confidence
            best_vision_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
            # Average confidence for that sentiment
            # (Simplification: just take average of all frames for now or just set high?)
            # Let's use the average confidence of the winning sentiment's frames
            winning_confs = [c for s, c in vision_results if s == best_vision_sentiment]
            avg_vision_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0.0
            
            results["vision"] = ModalityResult(sentiment=best_vision_sentiment, confidence=avg_vision_conf)
            
        # 3. Fusion Logic (Directly implementing weighted voting here)
        # Weights from fused_model.py: Text: 0.3, Audio: 0.35, Vision: 0.35
        weights = {"text": 0.3, "audio": 0.35, "vision": 0.35}
        
        final_scores = {}
        total_weight_used = 0.0
        
        for modality, result in results.items():
            s = result.sentiment
            c = result.confidence
            w = weights.get(modality, 0.33)
            
            # Map to unified sentiment
            unified_s = SENTIMENT_UNIFICATION.get(s, "Neutral")
            
            if unified_s not in final_scores:
                final_scores[unified_s] = 0.0
            final_scores[unified_s] += c * w
            total_weight_used += w
            
        if final_scores:
            final_sentiment = max(final_scores.items(), key=lambda x: x[1])[0]
            # Normalize confidence roughly
            final_confidence = final_scores[final_sentiment] / (total_weight_used if total_weight_used > 0 else 1.0)
            final_confidence = min(final_confidence, 1.0) # Cap at 1.0
        else:
            final_sentiment = "Unknown"
            final_confidence = 0.0
            
        return VideoResponse(
            video_sentiment=final_sentiment,
            video_confidence=final_confidence,
            modalities=results,
            frames_analyzed=len(frames),
            transcription=transcription
        )

    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(status_code=500, detail=str(e))
