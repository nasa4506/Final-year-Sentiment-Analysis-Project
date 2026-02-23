# Multimodal Sentiment Analysis Models Documentation

This document provides a detailed overview of the machine learning models used in the Multimodal Sentiment Analysis application. It covers text, audio, and vision modalities, including the specific model architectures utilized and references to their foundational research papers.

All reference research papers have been downloaded to the `/papers` directory in the project root.

---

## 1. Text Sentiment Analysis 📝

### Model Utilized
- **HuggingFace Hub ID:** `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Underlying Architecture:** RoBERTa (Robustly Optimized BERT Pretraining Approach)

### Description
The text modality employs a fine-tuned version of the **RoBERTa-base** model. RoBERTa builds upon BERT's language masking strategy, modifying key hyperparameters, removing the next-sentence pretraining objective, and training with much larger mini-batches and learning rates. This specific checkpoint is fine-tuned extensively on Twitter datasets for robust sentiment analysis, predicting `Negative`, `Neutral`, and `Positive` classes.

### Research Paper
- **Title:** *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- **Authors:** Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov (2019)
- **Local Location:** `./papers/RoBERTa_Paper.pdf`
- **Online PDF:** [arXiv:1907.11692](https://arxiv.org/pdf/1907.11692.pdf)

---

## 2. Audio/Speech Emotion Recognition 🎙️

### Model Utilized
- **HuggingFace Hub ID:** `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
- **Underlying Architecture:** Wav2Vec 2.0 (XLSR-53)

### Description
The audio component leverages the **Wav2Vec 2.0** architecture, specifically the Cross-Lingual Speech Representation (XLSR) model, to capture latent linguistic representations from raw audio waveforms. Wav2Vec 2.0 is highly effective at deriving self-supervised speech representations by masking the speech input and solving a contrastive task. This model processes 16kHz audio signals directly to determine emotional sentiment and pitch characteristics from vocal inflections, skipping traditional feature extraction pipelines (like MFCCs).

### Research Paper
- **Title:** *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*
- **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (2020)
- **Local Location:** `./papers/Wav2Vec2_Paper.pdf`
- **Online PDF:** [arXiv:2006.11477](https://arxiv.org/pdf/2006.11477.pdf)

---

## 3. Vision Sentiment Analysis 👁️

### Model Utilized
- **HuggingFace Hub ID:** `dima806/facial_emotions_image_detection`
- **Underlying Architecture:** Vision Transformer (ViT)
- *(Note: While the earlier project documentation `overview.md` references ResNet-50, the current backend implementation uses a fine-tuned Vision Transformer (ViT-base) model for facial expression recognition).*

### Description
The visual sentiment pipeline utilizes a **Vision Transformer (ViT)**. ViT models differ from standard Convolutional Neural Networks (CNNs) by treating an image as a sequence of patches, processed by a standard Transformer encoder. After initial face detection preprocessing (cropping the face tightly), the model embeds the 16x16 face image patches into sequences and classifies them into distinct emotional states. This ViT setup provides substantial improvements in capturing global context compared to traditional CNNs.

### Research Paper
- **Title:** *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
- **Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (2020)
- **Local Location:** `./papers/Vision_Transformer_ViT_Paper.pdf`
- **Online PDF:** [arXiv:2010.11929](https://arxiv.org/pdf/2010.11929.pdf)

---

## 4. Multimodal Fusion Strategy 🔗

### Implementation
- **Files:** `/backend/src/models/fused_model.py` and `routers/video.py`

### Description
The system provides two fusion strategies:
1. **Weighted Average (Late Fusion):** Sentiment scores from Text, Audio, and Vision are extracted individually. They are then mapped to unified labels (`Positive`, `Neutral`, `Negative`) and aggregated using a weighted average schema to produce a unified sentiment score.
2. **Max Fusion for Video:** Calculates individual modal sentiment timelines. Since audio typically shifts slower and texts are isolated snippets, the unified response takes the maximum confidence values across rolling windows.

This late-fusion pattern is widely supported by multi-modal deep learning literature as an effective technique for decoupled server architectures, ensuring robust predictions even if one modality is absent or noisy.
