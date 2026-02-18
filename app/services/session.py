from typing import Dict, List
from app.models.schemas import MessageItem

sessions_db: Dict[str, List[MessageItem]] = {}

class SessionManager:
    @staticmethod
    def add_message(session_id: str, role: str, content: str):
        if session_id not in sessions_db: sessions_db[session_id] = []
        sessions_db[session_id].append(MessageItem(role=role, content=content))

    @staticmethod
    def get_history(session_id: str) -> List[MessageItem]:
        return sessions_db.get(session_id, [])

    @staticmethod
    def get_observation_window(session_id: str, window=6) -> str:
        history = sessions_db.get(session_id, [])
        recent = history[-window:] if len(history) > window else history
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])