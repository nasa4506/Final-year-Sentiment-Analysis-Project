# Project Overview

## 1. Introduction
This project is a **Multimodal Sentiment Analysis** system. It is an evolution of a reference Streamlit application, refactored into a modern, decoupled **Client-Server Architecture**. The system analyzes sentiment from **Text**, **Audio**, and **Vision** (Image/Video) inputs, both individually and in a fused multi-modal manner.

## 2. Architecture

### Current Implementation (`/`)
-   **Type:** Decoupled Client-Server Application
-   **Backend:** FastAPI
-   **Frontend:** React (Vite + TypeScript)
-   **Core Logic:** Shared Python modules (`src/`) for model inference.
-   **Pros:** Scalable, clear separation of concerns, modern and responsive UI, RESTful API.

---

## 3. System Components

### 3.1. Backend (`/backend`)
The backend is built with **FastAPI**, serving as the inference engine and API layer.

-   **Entry Point:** `main.py` - Initialize API, CORS, and logging.
-   **Routers (`/backend/routers`):** Define REST endpoints for each modality:
    -   `text.py`: `/analyze/text` (Uses TextBlob)
    -   `audio.py`: `/analyze/audio` (Uses Wav2Vec2)
    -   `vision.py`: `/analyze/vision` (Uses ResNet-50)
    -   `fused.py`: `/analyze/fused` (Combines predictions)
    -   `video.py`: `/analyze/video` (Max Fusion strategy)
-   **Services (`/backend/services`):** Handles business logic like model loading (`model_loader.py`).

### 3.2. Frontend (`/frontend`)
The frontend is a modern Single Page Application (SPA) built with **React** and **Vite**.

-   **Tech Stack:**
    -   **Languages:** TypeScript, TSX
    -   **Styling:** Tailwind CSS (v4)
    -   **Icons:** Lucide React
    -   **Animations:** Framer Motion
    -   **Routing:** React Router DOM
    -   **HTTP Client:** Axios
-   **Key Features:**
    -   Responsive Design
    -   Real-time sentiment visualizers
    -   Interactive upload and recording interfaces

### 3.3. Core Logic (`/backend/src`)
This directory contains the central machine learning and processing logic, now consolidated within the backend.

-   **Models (`/backend/src/models`):** Wrappers for the AI models (Text, Audio, Vision, Fused).
-   **Utils (`/backend/src/utils`):** Helper functions for preprocessing (audio resampling, image resizing, video frame extraction).
-   **Config (`/backend/src/config`):** Configuration settings.

---

## 4. Feature Set

| Feature | Current (React + FastAPI) |
| :--- | :--- |
| **Text Analysis** | ✅ TextBlob via API |
| **Audio Analysis** | ✅ Wav2Vec2 (Upload/Record) |
| **Vision Analysis** | ✅ ResNet-50 (Upload/Camera) |
| **Fused Analysis** | ✅ Weighted Average via API |
| **Video Analysis** | ✅ Max Fusion via API |
| **UI/UX** | Custom Tailwind Components |

## 5. Directory Structure
```
/
├── backend/            # FastAPI Application & Core Logic
│   ├── routers/        # API Endpoints
│   ├── services/       # Business Logic
│   ├── src/            # Shared ML/Processing Core
│   │   ├── models/     # Model Inference Code
│   │   ├── utils/      # Preprocessing Utils
│   │   └── config/     # Settings
│   └── main.py         # App Entry Point
└── frontend/           # React Application
    ├── public/         # Static Assets
    └── src/            # React Components & Hooks
```
