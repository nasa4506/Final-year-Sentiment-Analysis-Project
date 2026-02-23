from backend.src.models.text_model import predict_text_sentiment
from backend.src.models.audio_model import predict_audio_sentiment
from backend.src.models.vision_model import predict_vision_sentiment
from backend.src.config.settings import SENTIMENT_UNIFICATION

def predict_fused_sentiment(text=None, audio_bytes=None, image=None):
    """
    Predict fused sentiment from available modalities.
    """
    results = []
    
    # 1. Text Analysis
    if text and text.strip():
        try:
            s, c, r = predict_text_sentiment(text)
            results.append({"modality": "text", "sentiment": s, "confidence": c, "reasoning": r})
        except Exception as e:
            print(f"Error in text fusion: {e}")

    # 2. Audio Analysis
    if audio_bytes:
        try:
            s, c = predict_audio_sentiment(audio_bytes)
            results.append({"modality": "audio", "sentiment": s, "confidence": c})
        except Exception as e:
            print(f"Error in audio fusion: {e}")

    # 3. Vision Analysis
    if image:
        try:
            s, c = predict_vision_sentiment(image)
            results.append({"modality": "vision", "sentiment": s, "confidence": c})
        except Exception as e:
            print(f"Error in vision fusion: {e}")

    if not results:
        return "Unknown", 0.0, [], []

    # Fusion Logic
    # Weights
    weights = {"text": 0.3, "audio": 0.35, "vision": 0.35}
    
    unified_scores = {"Positive": 0.0, "Negative": 0.0, "Neutral": 0.0}
    total_weight = 0.0
    math_breakdown = []
    
    for res in results:
        original_sentiment = res["sentiment"]
        confidence = res["confidence"]
        modality = res["modality"]
        
        # Map to unified sentiment
        unified_sentiment = SENTIMENT_UNIFICATION.get(original_sentiment, "Neutral")
        
        # Add weighted score
        weight = weights.get(modality, 0.33)
        contribution = confidence * weight
        unified_scores[unified_sentiment] += contribution
        total_weight += weight
        
        # Build Math Breakdown
        # Example format: "Text: Anticipation (82%) × Weight (0.3) = 0.246 Positive"
        math_breakdown.append({
            "modality": modality.capitalize(),
            "original_sentiment": original_sentiment.capitalize(),
            "confidence": round(confidence * 100, 1),
            "weight": weight,
            "contribution": round(contribution * 100, 1),
            "maps_to": unified_sentiment
        })
        
    # Find max score
    if total_weight > 0:
        # Normalize? No need if we just compare
        pass
        
    final_sentiment = max(unified_scores.items(), key=lambda x: x[1])[0]
    final_score = unified_scores[final_sentiment]
    
    # Normalize confidence to 0-1 range roughly
    if total_weight > 0:
        normalized_confidence = final_score / total_weight
    else:
        normalized_confidence = 0.0
        
    # Extract just text reasoning to return upwards
    text_reasoning = next((r["reasoning"] for r in results if r["modality"] == "text"), [])
        
    return final_sentiment, min(normalized_confidence, 1.0), text_reasoning, math_breakdown
