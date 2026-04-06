from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
from openai import OpenAI

from ml_model import get_regressor, get_extractor
from agents import get_agents_for_count, AGENTS

router = APIRouter()

# Initialize OpenAI client
openai_client = None
try:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"⚠️ OpenAI client init failed: {e}")


class PostRequest(BaseModel):
    postText: str
    userId: Optional[int] = 1


@router.post("/predict-score")
async def predict_score(request: PostRequest):
    """Predict engagement score for a post"""
    extractor = get_extractor()
    regressor = get_regressor()

    features_df = extractor.extract(request.postText)
    score = regressor.predict_score(features_df)
    agents_to_spawn = regressor.calculate_agents_to_spawn(score)

    return {
        "score": round(score, 2),
        "agentsToSpawn": agents_to_spawn,
        "maxAgents": len(AGENTS)
    }


@router.post("/get-responses")
async def get_responses(request: PostRequest):
    """Get AI agent responses for a post"""
    try:
        extractor = get_extractor()
        regressor = get_regressor()

        features_df = extractor.extract(request.postText)
        predicted_score = regressor.predict_score(features_df)
        agents_to_spawn = regressor.calculate_agents_to_spawn(predicted_score)
        agents_to_spawn = min(agents_to_spawn, len(AGENTS))

        agents = get_agents_for_count(agents_to_spawn)
        responses = []

        for agent in agents:
            try:
                if openai_client:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": agent.system_prompt},
                            {"role": "user", "content": f"Please respond to this post: {request.postText}"}
                        ],
                        max_tokens=150,
                        temperature=0.7
                    )
                    agent_response = response.choices[0].message.content
                else:
                    agent_response = f"[{agent.name}]: I find this topic interesting! (OpenAI not configured)"
            except Exception as e:
                print(f"Error getting response from {agent.name}: {e}")
                agent_response = f"[{agent.name}]: Interesting point! (API error)"

            responses.append({
                "agentId": agent.id,
                "agentName": agent.name,
                "agentPersona": agent.persona,
                "color": agent.color,
                "emoji": agent.emoji,
                "response": agent_response
            })

        return {
            "predictedScore": round(predicted_score, 2),
            "agentsSpawned": len(responses),
            "responses": responses
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
