import torch
import numpy as np
from scipy.special import softmax
from backend.services.model_loader import model_loader
from backend.src.utils.preprocessing import preprocess_audio_for_model

def predict_audio_sentiment(audio_bytes: bytes):
    """
    Predict sentiment from audio bytes using Wav2Vec2 model.
    """
    # Load model
    model, device, config, feature_extractor = model_loader.load_audio_model()
    
    # Preprocess audio
    input_values = preprocess_audio_for_model(audio_bytes)
    
    if input_values is None:
        return "Error", 0.0
        
    input_values = input_values.to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_values)
    
    # Post-processing
    scores = output.logits[0].detach().cpu().numpy()
    scores = softmax(scores)
    
    # Get ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    # Map to labels
    # Use model's internal config for labels
    id2label = config.id2label
    
    top_label_idx = ranking[0]
    top_score = scores[top_label_idx]
    
    sentiment = id2label[top_label_idx]
    confidence = float(top_score)
    
    return sentiment, confidence
