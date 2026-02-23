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
This project uses advanced deep learning models for sentiment analysis:
- **Text:** RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Audio:** Wav2Vec 2.0 (`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`)
- **Vision:** Vision Transformer - ViT (`dima806/facial_emotions_image_detection`)

For more details on the architecture and models, refer to `models_documentation.md` and `overview.md`.

## Troubleshooting
- **CORS Errors:** If the frontend cannot communicate with the backend, ensure the backend is running and the CORS middleware in `backend/main.py` allows origins from your Vite dev server port.
- **Missing Models:** The models are downloaded automatically via HuggingFace the first time they are used. Ensure you have an active internet connection on your first run.
