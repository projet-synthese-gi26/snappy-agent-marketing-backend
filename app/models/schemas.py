from pydantic import BaseModel
from typing import List, Optional, Dict

class MessageItem(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    action_taken: str
    belief_state: Dict[str, float] # On renvoie les probas pour le debug
    history: List[MessageItem]