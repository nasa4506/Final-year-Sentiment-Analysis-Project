# Backend Architecture

## 1. Overview
The backend is a **FastAPI** application designed to serve as the inference engine for the Multimodal Sentiment Analysis system. It exposes RESTful endpoints for Text, Audio, and Vision sentiment analysis, as well as a Fused analysis mode. The architecture emphasizes separation of concerns, utilizing a singleton pattern for efficient model loading and management.

## 2. Core Components

### 2.1. Entry Point (`main.py`)
-   Initializes the FastAPI app.
-   Configures CORS (Cross-Origin Resource Sharing).
-   Sets up logging.
-   Includes routers for different modalities.

### 2.2. Service Layer (`services/`)
-   **`ModelLoader` (`services/model_loader.py`):** A **Singleton** class responsible for loading and caching heavy machine learning models. It ensures models are loaded only once and shared across requests.
    -   **Responsibility:** Load Tokenizers, Feature Extractors, and Model Weights from Hugging Face.
    -   **Device Management:** Automatically selects CUDA (GPU) if available, falling back to CPU.

### 2.3. API Layer (`routers/`)
-   **`text.py`:** Handles text analysis requests.
-   **`audio.py`:** Handles audio file uploads and processing.
-   **`vision.py`:** Handles image uploads and processing.
-   **`fused.py`:** Handles multi-modal requests, aggregating results from individual models.
-   **`video.py`:** Implements "Max Fusion" strategy for video files.

### 2.4. Inference Layer (`src/models/`)
Contains the specific inference logic for each modality. These modules use the `ModelLoader` to access the loaded models.
-   **`text_model.py`:** Uses `twitter-roberta-base-sentiment-latest` for text sentiment.
-   **`audio_model.py`:** Uses `wav2vec2-lg-xlsr-en-speech-emotion-recognition` for audio sentiment.
-   **`vision_model.py`:** Uses `facial_emotions_image_detection` (ViT-based) for facial expression analysis.

## 3. Data Flow

1.  **Request:** Client sends data (JSON text or File) to an endpoint (e.g., `/analyze/text`).
2.  **Router:** The route handler receives the request and validates the input.
3.  **Service Access:** The handler calls the specific prediction function from `src/models/`.
4.  **Model Retrieval:** The prediction function requests the model from `ModelLoader`.
5.  **Inference:**
    -   **Preprocessing:** Input is processed (tokenized, resized, resampled) using handlers in `src/utils/` or Hugging Face processors.
    -   **Prediction:** The model generates logits/probabilities.
    -   **Post-processing:** Logits are converted to human-readable labels (Positive, Negative, Neutral) and confidence scores.
6.  **Response:** The result is returned to the client as a JSON response.

## 4. Model Integration (Hugging Face)

We are replacing mock data and Google Drive dependencies with direct Hugging Face integration.

| Modality | Model ID | Type |
| :--- | :--- | :--- |
| **Text** | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Transformer (RoBERTa) |
| **Audio** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Wav2Vec2 |
| **Vision** | `dima806/facial_emotions_image_detection` | Vision Transformer (ViT) |

## 5. Directory Structure
```
backend/
├── main.py
├── architecture.md
├── routers/
│   ├── text.py
│   ├── audio.py
│   ├── vision.py
│   ├── fused.py
│   └── video.py
└── services/
    └── model_loader.py  <-- Singleton Model Manager
src/
├── config/
│   └── settings.py      <-- Model IDs and Configs
├── models/
│   ├── text_model.py
│   ├── audio_model.py
│   └── vision_model.py
└── utils/
    └── preprocessing.py <-- Shared preprocessing logic
```
