from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal Sentiment Analysis API",
    description="API for analyzing sentiment from Text, Audio, and Vision inputs.",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, allow all. restricted in prod.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Multimodal Sentiment Analysis API is running", "status": "ok"}

from backend.routers import text, audio, vision, fused, video

app.include_router(text.router)
app.include_router(audio.router)
app.include_router(vision.router)
app.include_router(fused.router)
app.include_router(video.router)
