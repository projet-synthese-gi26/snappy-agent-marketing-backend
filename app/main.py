import uuid
import numpy as np
from pydantic import BaseModel
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, MessageItem, ListenResponse
from app.services.config import load_product_config
from app.core.agent import MarketingAgent
from app.services.session import SessionManager

SessionManager.init_db()
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
    agent = MarketingAgent(product)
except Exception as e:
    print(f"Erreur démarrage: {e}")
    exit(1)


def get_session_belief_db(sid: str) -> np.ndarray:
    stored = SessionManager.get_belief_state(sid)
    if stored is not None:
        return stored
    return agent.belief_tracker.get_initial_belief()

def update_session_belief_db(sid: str, vec: np.ndarray):
    SessionManager.save_belief_state(sid, vec)

# --- NOUVEAUX SCHÉMAS ---
class HumanMessageRequest(BaseModel):
    session_id: str
    role: str # Doit être 'user' ou 'agent'
    content: str


# --- ENDPOINTS ---

@app.get("/chat/{session_id}/history", response_model=list[MessageItem])
async def get_history_endpoint(session_id: str):
    return SessionManager.get_history(session_id)


@app.post("/message")
async def send_human_message(request: HumanMessageRequest):
    """Enregistre un message en BD (Mode OFF/Humain) sans déclencher l'IA."""
    SessionManager.add_message(request.session_id, request.role, request.content)
    return {"status": "ok"}


@app.post("/agent/listen", response_model=ListenResponse)
async def agent_listen(request: ChatRequest):
    """Mode LISTEN: Analyse le dernier message et met à jour le Belief State, silencieusement."""
    sid = request.session_id
    res = agent.process_listen_turn(
        last_user_utterance=request.message,
        belief_manager=(get_session_belief_db, update_session_belief_db),
        session_id=sid
    )
    return ListenResponse(status="ok", belief_state=res["belief"].probabilities)


@app.post("/agent/respond", response_model=ChatResponse)
async def agent_respond(request: ChatRequest):
    """Mode ON: L'agent lit l'historique et génère la réponse."""
    sid = request.session_id
    obs = SessionManager.get_observation_window(sid)
    res = agent.process_turn_for_api(
        history_observation=obs,
        last_user_utterance=request.message,
        belief_manager=(get_session_belief_db, update_session_belief_db),
        session_id=sid
    )
    SessionManager.add_message(sid, "agent", res["reply"])
    return ChatResponse(
        session_id=sid,
        reply=res["reply"],
        action_taken=res["action"].name,
        belief_state=res["belief"].probabilities,
        history=SessionManager.get_history(sid)
    )

# Gardé pour rétrocompatibilité avec tes anciens scripts (benchmark)
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    sid = request.session_id or str(uuid.uuid4())
    SessionManager.add_message(sid, "user", request.message)
    obs = SessionManager.get_observation_window(sid)
    res = agent.process_turn_for_api(
        history_observation=obs,
        last_user_utterance=request.message,
        belief_manager=(get_session_belief_db, update_session_belief_db),
        session_id=sid
    )
    SessionManager.add_message(sid, "agent", res["reply"])
    return ChatResponse(session_id=sid, reply=res["reply"], action_taken=res["action"].name, belief_state=res["belief"].probabilities, history=[])

@app.post("/listen", response_model=ListenResponse)
async def listen_endpoint(request: ChatRequest):
    sid = request.session_id or str(uuid.uuid4())
    SessionManager.add_message(sid, "user", request.message)
    res = agent.process_listen_turn(last_user_utterance=request.message, belief_manager=(get_session_belief_db, update_session_belief_db), session_id=sid)
    return ListenResponse(status="ok", belief_state=res["belief"].probabilities)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)