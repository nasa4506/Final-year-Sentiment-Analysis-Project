# Text Model Configuration (Multilingual 13-Emotions via LoRA)
TEXT_MODEL_CONFIG = {
    # The base model is what the tokenzier needs
    "base_model_name": "xlm-roberta-base",
    # The local path where our fine-tuned LoRA adapters are saved
    "lora_path": "./models_weights/multilingual_13_emotions_lora",
    "labels": [
        "angry", "anticipation", "confusion", "disgust", "fear", "frustration", 
        "happy", "joy", "love", "neutral", "sad", "surprise", "trust"
    ]
}

# Audio Model Configuration
AUDIO_MODEL_CONFIG = {
    "model_name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
    "target_sampling_rate": 16000,
    "max_duration": 5  # seconds
}

# Vision Model Configuration
VISION_MODEL_CONFIG = {
    "model_name": "dima806/facial_emotions_image_detection",
    "image_size": 224,
    "normalize_mean": [0.5, 0.5, 0.5], # ViT usually uses 0.5 mean/std or ImageNet
    "normalize_std": [0.5, 0.5, 0.5]
}

# Image Preprocessing Configuration
IMAGE_TRANSFORMS = {
    "resize": 224,
    "center_crop": 224,
    "normalize_mean": [0.485, 0.456, 0.406], # ImageNet defaults, kept for face extraction/ResNet compat if needed
    "normalize_std": [0.229, 0.224, 0.225],
}

# File Processing Configuration
SUPPORTED_IMAGE_FORMATS = ["png", "jpg", "jpeg", "bmp", "tiff"]
SUPPORTED_AUDIO_FORMATS = ["wav", "mp3", "m4a", "flac"]
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "wmv", "flv"]

# Video Processing Configuration
MAX_VIDEO_FRAMES = 5
VIDEO_FRAME_INTERVALS = [0, 0.25, 0.5, 0.75, 1.0]

# Sentiment Grouping (for fusion)
SENTIMENT_UNIFICATION = {
    # Base
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
    
    # Audio/Vision Original
    "Happy": "Positive",
    "Sad": "Negative",
    "Angry": "Negative",
    "Fear": "Negative",
    "Disgust": "Negative",
    "Surprise": "Neutral",
    
    # Text Model 13 Emotions
    "joy": "Positive",
    "love": "Positive",
    "happy": "Positive",
    "anticipation": "Positive",  # Generally positive/excitement
    "trust": "Positive",
    
    "neutral": "Neutral",
    "confusion": "Neutral",
    "surprise": "Neutral",
    
    "sad": "Negative",
    "angry": "Negative",
    "fear": "Negative",
    "disgust": "Negative",
    "frustration": "Negative"
}

