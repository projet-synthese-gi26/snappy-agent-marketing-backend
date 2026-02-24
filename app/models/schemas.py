from pydantic import BaseModel
from typing import List, Optional, Dict

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    context_window: Optional[int] = 6  # Nouvelle propriété pour la fenêtre de contexte

class HumanMessageRequest(BaseModel):
    session_id: Optional[str] = None
    role: str
    content: str

class HumanMessageResponse(BaseModel):
    status: str
    session_id: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    action_taken: str
    belief_state: Dict[str, float]
    history: List[MessageItem]

class ListenResponse(BaseModel):
    status: str
    session_id: str
    belief_state: Dict[str, float]