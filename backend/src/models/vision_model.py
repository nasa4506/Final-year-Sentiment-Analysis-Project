import torch
import numpy as np
from scipy.special import softmax
from PIL import Image
from backend.services.model_loader import model_loader
from backend.src.utils.preprocessing import detect_and_preprocess_face

def predict_vision_sentiment(image, crop_tightness=0.0):
    """
    Predict sentiment from image using ViT model.
    
    Args:
        image: PIL Image or numpy array
        crop_tightness: Float, padding for face crop
    """
    # Load model
    model, device, config, processor = model_loader.load_vision_model()
    
    # Preprocess image (Face detection + crop)
    # The new model expects 224x224, but the processor handles resizing.
    # However, face detection is crucial.
    face_image = detect_and_preprocess_face(image, crop_tightness)
    
    if face_image is None:
        # If face detection fails, use original image (fallback is inside detect_and_preprocess_face but let's be safe)
        face_image = image
    
    # helper to ensure we have a PIL image
    if not isinstance(face_image, Image.Image):
        try:
             face_image = Image.fromarray(face_image)
        except:
             pass

    # Prepare input for model
    try:
        inputs = processor(images=face_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            output = model(**inputs)
        
        # Post-processing
        scores = output.logits[0].detach().cpu().numpy()
        scores = softmax(scores)
        
        # Get ranking
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        # Map to labels
        id2label = config.id2label
        
        top_label_idx = ranking[0]
        top_score = scores[top_label_idx]
        
        sentiment = id2label[top_label_idx]
        confidence = float(top_score)
        
        return sentiment, confidence
    except Exception as e:
        print(f"Error in vision prediction: {e}")
        return "Error", 0.0
