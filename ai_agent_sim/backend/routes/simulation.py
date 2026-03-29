from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List
import json
import asyncio
from database import get_db
from models import Simulation, AgentResponse, User
from ml_model import get_model
from agents import get_agents_for_count, AGENTS
import os
from openai import OpenAI

router = APIRouter()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PostRequest(BaseModel):
    postText: str
    userId: Optional[int] = 1  # Default for testing

class SimulationResponse(BaseModel):
    id: int
    postText: str
    predictedScore: float
    agentsSpawned: int
    responses: List[dict]

@router.post("/predict-score")
async def predict_score(request: PostRequest, db: Session = Depends(get_db)):
    """Predict engagement score for a post"""
    model = get_model()
    score = model.predict_score(request.postText)
    agents_to_spawn = model.calculate_agents_to_spawn(score)
    
    return {
        "score": round(score, 2),
        "agentsToSpawn": agents_to_spawn,
        "maxAgents": len(AGENTS)
    }

@router.post("/start")
async def start_simulation(request: PostRequest, db: Session = Depends(get_db)):
    """Start a new simulation with dynamic agent spawning"""
    try:
        # Predict score and determine agent count
        model = get_model()
        predicted_score = model.predict_score(request.postText)
        agents_to_spawn = model.calculate_agents_to_spawn(predicted_score)
        
        # Ensure we don't spawn more agents than available
        agents_to_spawn = min(agents_to_spawn, len(AGENTS))
        
        # Create simulation record
        simulation = Simulation(
            userId=request.userId,
            postText=request.postText,
            predictedScore=predicted_score,
            agentsSpawned=agents_to_spawn
        )
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        return {
            "simulationId": simulation.id,
            "predictedScore": round(predicted_score, 2),
            "agentsToSpawn": agents_to_spawn,
            "message": f"Simulation started. {agents_to_spawn} agents will respond."
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-responses")
async def get_responses(request: PostRequest, db: Session = Depends(get_db)):
    """Get AI agent responses for a post"""
    try:
        # Predict score and determine agent count
        model = get_model()
        predicted_score = model.predict_score(request.postText)
        agents_to_spawn = model.calculate_agents_to_spawn(predicted_score)
        agents_to_spawn = min(agents_to_spawn, len(AGENTS))
        
        # Get agents to respond
        agents = get_agents_for_count(agents_to_spawn)
        
        responses = []
        
        for agent in agents:
            try:
                # Generate response using OpenAI
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": agent.system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Please respond to this post: {request.postText}"
                        }
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                
                agent_response = response.choices[0].message.content
                
                responses.append({
                    "agentId": agent.id,
                    "agentName": agent.name,
                    "agentPersona": agent.persona,
                    "color": agent.color,
                    "emoji": agent.emoji,
                    "response": agent_response
                })
                
            except Exception as e:
                print(f"Error getting response from {agent.name}: {e}")
                responses.append({
                    "agentId": agent.id,
                    "agentName": agent.name,
                    "agentPersona": agent.persona,
                    "color": agent.color,
                    "emoji": agent.emoji,
                    "response": f"[Error generating response: {str(e)[:50]}...]"
                })
        
        return {
            "predictedScore": round(predicted_score, 2),
            "agentsSpawned": len(responses),
            "responses": responses
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/simulations/{simulation_id}")
async def get_simulation(simulation_id: int, db: Session = Depends(get_db)):
    """Get a specific simulation with all responses"""
    simulation = db.query(Simulation).filter(Simulation.id == simulation_id).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    responses = db.query(AgentResponse).filter(
        AgentResponse.simulationId == simulation_id
    ).all()
    
    return {
        "id": simulation.id,
        "postText": simulation.postText,
        "predictedScore": simulation.predictedScore,
        "agentsSpawned": simulation.agentsSpawned,
        "createdAt": simulation.createdAt.isoformat(),
        "responses": [
            {
                "agentName": r.agentName,
                "agentPersona": r.agentPersona,
                "response": r.response
            }
            for r in responses
        ]
    }

@router.get("/simulations")
async def list_simulations(user_id: int = 1, db: Session = Depends(get_db)):
    """List all simulations for a user"""
    simulations = db.query(Simulation).filter(
        Simulation.userId == user_id
    ).order_by(Simulation.createdAt.desc()).all()
    
    return {
        "count": len(simulations),
        "simulations": [
            {
                "id": s.id,
                "postText": s.postText[:100] + "..." if len(s.postText) > 100 else s.postText,
                "predictedScore": s.predictedScore,
                "agentsSpawned": s.agentsSpawned,
                "createdAt": s.createdAt.isoformat()
            }
            for s in simulations
        ]
    }
