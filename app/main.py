import uuid
import numpy as np
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse
from app.services.config import load_product_config
from app.core.agent import MarketingAgent
from app.services.session import SessionManager

app = FastAPI(title="Snappy Agent (Groq Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Chargement Produit
try:
    product = load_product_config("product_config.json")
    # Instanciation de l'agent (Groq par défaut via core/llm.py)
    agent = MarketingAgent(product)
except Exception as e:
    print(f"Erreur démarrage: {e}")
    exit(1)

# 2. Gestion Session
session_beliefs: Dict[str, np.ndarray] = {}

def get_session_belief(sid: str) -> np.ndarray:
    if sid not in session_beliefs:
        session_beliefs[sid] = agent.belief_tracker.get_initial_belief()
    return session_beliefs[sid]

def update_session_belief(sid: str, vec: np.ndarray):
    session_beliefs[sid] = vec

# 3. Endpoint Chat
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    sid = request.session_id or str(uuid.uuid4())
    SessionManager.add_message(sid, "user", request.message)
    
    obs = SessionManager.get_observation_window(sid)
    
    # Appel Agent
    res = agent.process_turn_for_api(
        history_observation=obs,
        last_user_utterance=request.message,
        belief_manager=(get_session_belief, update_session_belief),
        session_id=sid
    )
    
    SessionManager.add_message(sid, "agent", res["reply"])
    
    # Dans un vrai article, on loggerait res["reward"] ici
    
    return ChatResponse(
        session_id=sid,
        reply=res["reply"],
        action_taken=res["action"].name,
        belief_state=res["belief"].probabilities,
        history=SessionManager.get_history(sid)
    )

if __name__ == "__main__":
    import uvicorn
    # Lancement pour le debug
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)