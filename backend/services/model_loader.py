import logging
import torch
from transformers import AutoConfig, AutoModelForAudioClassification, AutoFeatureExtractor
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.src.config.settings import AUDIO_MODEL_CONFIG, VISION_MODEL_CONFIG, TEXT_MODEL_CONFIG

logger = logging.getLogger(__name__)

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.audio_model = None
            cls._instance.vision_model = None
            cls._instance.text_model = None
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return cls._instance

    def load_audio_model(self):
        if self.audio_model is None:
            try:
                logger.info("Loading Audio Model...")
                model_name = AUDIO_MODEL_CONFIG["model_name"]
                self.audio_config = AutoConfig.from_pretrained(model_name)
                self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
                self.audio_model = AutoModelForAudioClassification.from_pretrained(model_name)
                self.audio_model.to(self.device)
                logger.info(f"Audio model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load audio model: {e}")
                raise e
        return self.audio_model, self.device, self.audio_config, self.audio_feature_extractor

    def load_vision_model(self):
        if self.vision_model is None:
            try:
                logger.info("Loading Vision Model...")
                model_name = VISION_MODEL_CONFIG["model_name"]
                self.vision_config = AutoConfig.from_pretrained(model_name)
                self.vision_processor = AutoImageProcessor.from_pretrained(model_name)
                self.vision_model = AutoModelForImageClassification.from_pretrained(model_name)
                self.vision_model.to(self.device)
                logger.info(f"Vision model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load vision model: {e}")
                raise e
        return self.vision_model, self.device, self.vision_config, self.vision_processor

    def load_text_model(self):
        if self.text_model is None:
            try:
                logger.info("Loading Text Model...")
                model_name = TEXT_MODEL_CONFIG["model_name"]
                self.text_config = AutoConfig.from_pretrained(model_name)
                self.text_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.text_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.text_model.to(self.device)
                logger.info(f"Text model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load text model: {e}")
                raise e
        return self.text_model, self.device, self.text_config, self.text_tokenizer

model_loader = ModelLoader()
