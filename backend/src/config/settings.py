# Text Model Configuration
TEXT_MODEL_CONFIG = {
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "labels": ["Negative", "Neutral", "Positive"]
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
    "Positive": "Positive",
    "Happy": "Positive",
    "Negative": "Negative",
    "Sad": "Negative",
    "Angry": "Negative",
    "Fear": "Negative",
    "Disgust": "Negative",
    "Neutral": "Neutral",
    "Surprise": "Neutral"
}

