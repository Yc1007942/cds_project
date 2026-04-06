from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (moltnet/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Import routers
from routes import auth, simulation, data, inference

# Lifespan context — preload models on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 FastAPI server starting...")
    print(f"   Project root: {PROJECT_ROOT}")

    # Preload data + models in background
    from ml_model import get_features_df, get_classifier, get_regressor, get_extractor
    get_features_df()
    get_regressor()
    get_classifier()
    get_extractor()

    print("✅ All models and data loaded")
    yield
    print("🛑 FastAPI server shutting down...")

# Create FastAPI app
app = FastAPI(
    title="MoltNet — AI Agent Sprite Simulation",
    description="Neural operations deck with AI agent simulation, data exploration, and live inference",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "moltnet-ai-sim"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
