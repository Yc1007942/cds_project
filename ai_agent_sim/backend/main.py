from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from routes import auth, simulation

# Lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 FastAPI server starting...")
    yield
    # Shutdown
    print("🛑 FastAPI server shutting down...")

# Create FastAPI app
app = FastAPI(
    title="AI Agent Sprite Simulation",
    description="Dynamic AI agent discussion simulator with regression-based engagement",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["simulation"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ai-agent-sprite-sim"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
