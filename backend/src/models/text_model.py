import torch
import numpy as np
from scipy.special import softmax
from backend.services.model_loader import model_loader
from backend.src.config.settings import TEXT_MODEL_CONFIG

def predict_text_sentiment(text: str):
    """
    Predict sentiment from text using RoBERTa model.
    """
    # Load model and tokenizer
    model, device, config, tokenizer = model_loader.load_text_model()
    
    # Preprocess text
    encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    # Inference
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Post-processing
    scores = output.logits[0].detach().cpu().numpy()
    scores = softmax(scores)
    
    # Get ranking
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    
    # Map to labels
    # config.id2label might be available, but let's use our settings to be safe/consistent
    labels = TEXT_MODEL_CONFIG["labels"]
    
    top_label_idx = ranking[0]
    top_score = scores[top_label_idx]
    
    # Handle potentially different label mappings if needed, 
    # but initially assuming 0: Negative, 1: Neutral, 2: Positive
    # If the model uses a different mapping, we should adjust in settings.
    # The selected model cardiffnlp/twitter-roberta-base-sentiment-latest
    # usually has 0: negative, 1: neutral, 2: positive.
    
    sentiment = labels[top_label_idx]
    confidence = float(top_score)
    
    return sentiment, confidence
