import uuid
import numpy as np
from typing import Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import (
    ChatRequest, ChatResponse, MessageItem, 
    ListenResponse, HumanMessageRequest, HumanMessageResponse
)
from app.services.config import load_product_config
from app.core.agent import MarketingAgent
from app.services.session import SessionManager

SessionManager.init_db()
app = FastAPI(title="Snappy Agent (Production)")

# Sécurité: En production, on restreint les origines
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

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


# --- ENDPOINTS PRODUCTION ---

@app.get("/healthcheck")
async def health_check():
    """Vérifie que l'API est opérationnelle."""
    return {"status": "ok"}


@app.get("/chat/{session_id}/history", response_model=list[MessageItem])
async def get_history_endpoint(session_id: str):
    """Récupère l'historique. Retourne vide si la session n'existe pas."""
    if not session_id or session_id == "null":
        return []
    return SessionManager.get_history(session_id)


@app.post("/message", response_model=HumanMessageResponse)
async def send_human_message(request: HumanMessageRequest):
    """Enregistre un message humain. Crée une session si elle n'existe pas."""
    sid = request.session_id or str(uuid.uuid4())
    SessionManager.add_message(sid, request.role, request.content)
    return HumanMessageResponse(status="ok", session_id=sid)


@app.post("/agent/listen", response_model=ListenResponse)
async def agent_listen(request: ChatRequest):
    """Met à jour le Belief State sans répondre."""
    sid = request.session_id or str(uuid.uuid4())
    res = agent.process_listen_turn(
        last_user_utterance=request.message,
        belief_manager=(get_session_belief_db, update_session_belief_db),
        session_id=sid
    )
    return ListenResponse(status="ok", session_id=sid, belief_state=res["belief"].probabilities)


@app.post("/agent/respond", response_model=ChatResponse)
async def agent_respond(request: ChatRequest):
    """L'agent lit la fenêtre de contexte demandée et génère la réponse."""
    sid = request.session_id or str(uuid.uuid4())
    window_size = request.context_window or 6
    
    # On passe la taille de la fenêtre de contexte !
    obs = SessionManager.get_observation_window(sid, window=window_size)
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)