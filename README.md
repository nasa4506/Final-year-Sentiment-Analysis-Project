# Multimodal Sentiment Analysis

This project is a modern, decoupled Client-Server application for analyzing sentiment across Text, Audio, and Vision (Image/Video) modalities. It uses a FastAPI backend for inference and a React (Vite) frontend for an interactive user interface.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Python 3.8+** (for the backend)
- **Node.js 18+** (for the frontend)
- **npm** or **yarn** (Node.js package manager)
- **Git** (optional, for cloning the repository)

## Project Structure

```text
/
├── backend/            # FastAPI Application & Core Logic
│   ├── routers/        # API Endpoints
│   ├── services/       # Business Logic
│   ├── src/            # Shared ML/Processing Core
│   └── main.py         # App Entry Point
└── frontend/           # React Application
    ├── public/         # Static Assets
    └── src/            # React Components & Hooks
```

---

## 🚀 Setting up the Backend

The backend serves as the inference engine and exposes RESTful API endpoints.

1. **Navigate to the corresponding directory:**
   ```bash
   cd "g:/Final year project/backend"
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - **Windows:**
     ```bash
     .\venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install the required Python packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: The first run might take a few minutes as it downloads large ML models (PyTorch, Transformers, Wav2Vec2, ViT).*

5. **Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   The API will be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

---

## 🎨 Setting up the Frontend

The frontend provides the interactive User Interface built with React, Vite, and Tailwind CSS.

1. **Open a new terminal window/tab.**

2. **Navigate to the frontend directory:**
   ```bash
   cd "g:/Final year project/frontend"
   ```

3. **Install the Node.js dependencies:**
   ```bash
   npm install
   ```

4. **Start the development server:**
   ```bash
   npm run dev
   ```

5. **Access the application:**
   Open your browser and navigate to the URL provided in the terminal (usually `http://localhost:5173`).

---

## 🧠 Models Used
This project uses advanced deep learning models for multimodal sentiment analysis:
- **Text (English):** RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Text (Hindi):** Fine-tuned `l3cube-pune/hindi-bert-v2` trained on a custom combined dataset of Amazon Reviews and the Kaggle Hindi Emotion dataset (`praths71018/hindi-sentiment-dataset`, 13 emotions mapped to Positive/Neutral/Negative).
- **Audio:** Wav2Vec 2.0 (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`)
- **Vision:** Vision Transformer - ViT (`dima806/facial_emotions_image_detection`)

For more details on the architecture and models, refer to `models_documentation.md` and `overview.md`.

---

## 🏗️ Development Journey & Architectural Evolution

This project has undergone a significant transformation from its initial prototype stage to a modern, highly optimized Client-Server application. Here is a detailed breakdown of the development process:

### 1. Architectural Refactoring (Streamlit to React + FastAPI)
The initial prototype was built using Streamlit, which, while excellent for rapid iterations, posed limitations in scalability and complex UI design constraints.
- **Backend Transition**: We extracted the core machine-learning logic into an asynchronous **FastAPI** application. This allowed us to expose highly performant REST endpoints for Text, Audio, Vision, Video, and Fused analysis individually.
- **Frontend Transition**: We built a fully custom frontend Single Page Application (SPA) using **React, Vite, TypeScript, and Tailwind CSS**. This gave us granular control over the user interface, resulting in smooth animations (using Framer Motion), dynamic visualizations, and a highly responsive multi-modal dashboard.

### 2. Upgrading to State-of-the-Art (SOTA) Transformer Models
We moved away from legacy, primitive models (like TextBlob and lightweight ResNets) to leverage the power of advanced Transformers via HuggingFace:
- **English Text**: Upgraded to a RoBERTa architecture (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Audio**: Implemented a robust Wav2Vec 2.0 model for granular Speech Emotion Recognition.
- **Vision/Facial Cues**: Adopted a Vision Transformer (ViT) architecture (`dima806/facial_emotions_image_detection`) capable of nuanced facial expression analysis.

### 3. Curating and Training a Custom Hindi Sentiment AI 
A core focus of this project was addressing the lack of robust, localized emotion models in the Hindi language.
- **Dataset Curation**: We mined the Kaggle Hindi Emotion dataset (which contained 13 specific emotions like Joy, Love, Disgust, etc.) and programmatically mapped those down to the standard 3-class sentiment scale (Positive, Neutral, Negative).
- **Data Merging**: To ensure the model generalizes perfectly to real-world vocabulary structures, we merged the refined Kaggle dataset with a large Amazon Hindi Reviews dataset, establishing a highly robust, unified training set.
- **Fine-Tuning Process**: We fine-tuned the `l3cube-pune/hindi-bert-v2` transformer backbone using the `transformers` library on this custom dataset.
- **Hardware Optimization**: To facilitate local training under major hardware constraints (an Nvidia RTX 3050 mobile GPU with precisely 4GB of VRAM), we employed heavy VRAM optimization techniques: mixed-precision training (`fp16=True`), strict tokenizer truncation (`MAX_LENGTH=128`), an incredibly small physical batch size of 4, and utilized `gradient_accumulation_steps=4` to safely simulate a larger effective batch size of 16 without triggering CUDA Out-Of-Memory (OOM) errors.

### 4. Implementing Explainable AI (XAI) and Math Unification
To make the AI decisions deeply transparent to the user, we built Explainability directly into the application's core logic.
- **Token-Level Attributions**: Within the text output stream on the UI, individual words and subword tokens are distinctly highlighted based on their mathematical weights. Green glows indicate support for the predicted emotion, and red glows represent contradictions over the overall sentiment—with glow intensity tied directly to its impact score.
- **Math Breakdown Module**: We created an advanced tabular Evidence Panel that breaks down how every active modality (Audio, Vision, Text) unifies its specialized prediction string to a base sentiment. It demonstrates exactly how its internal confidence metric multiplies a predefined backend weight to yield a tangible total percentage contribution score applied to the final Fused output.

---

## 🛠 Troubleshooting
- **CORS Errors:** If the frontend cannot communicate with the backend, ensure the backend is running and the CORS middleware in `backend/main.py` allows origins from your Vite dev server port.
- **Missing Models:** The models are downloaded automatically via HuggingFace the first time they are used. Ensure you have an active internet connection on your first run.
