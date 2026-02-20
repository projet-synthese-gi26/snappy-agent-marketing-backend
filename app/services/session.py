import sqlite3
import json
import numpy as np
from typing import List, Dict, Any
from app.models.schemas import MessageItem

DB_PATH = "snappy_storage.db"

class SessionManager:
    @staticmethod
    def _get_conn():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def init_db():
        """Initialise les tables si elles n'existent pas."""
        conn = SessionManager._get_conn()
        cursor = conn.cursor()
        
        # Table Historique des Messages
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table État de Croyance (Pour le POMDP)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_states (
                session_id TEXT PRIMARY KEY,
                belief_vector TEXT NOT NULL, -- Stocké en JSON
                last_interaction_time REAL
            )
        """)
        
        conn.commit()
        conn.close()

    # --- GESTION DES MESSAGES ---

    @staticmethod
    def add_message(session_id: str, role: str, content: str):
        conn = SessionManager._get_conn()
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        conn.commit()
        conn.close()

    @staticmethod
    def get_history(session_id: str) -> List[MessageItem]:
        conn = SessionManager._get_conn()
        cursor = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        return [MessageItem(role=row["role"], content=row["content"]) for row in rows]

    @staticmethod
    def get_observation_window(session_id: str, window=6) -> str:
        """Récupère les N derniers messages pour le contexte LLM."""
        history = SessionManager.get_history(session_id)
        recent = history[-window:] if len(history) > window else history
        return "\n".join([f"{msg.role}: {msg.content}" for msg in recent])

    # --- GESTION DU BELIEF STATE (PERSISTANCE POMDP) ---

    @staticmethod
    def save_belief_state(session_id: str, belief_vec: np.ndarray):
        conn = SessionManager._get_conn()
        # On sérialise le numpy array en liste JSON
        vec_json = json.dumps(belief_vec.tolist())
        
        conn.execute("""
            INSERT INTO session_states (session_id, belief_vector) 
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET belief_vector = excluded.belief_vector
        """, (session_id, vec_json))
        conn.commit()
        conn.close()

    @staticmethod
    def get_belief_state(session_id: str) -> np.ndarray:
        conn = SessionManager._get_conn()
        cursor = conn.execute("SELECT belief_vector FROM session_states WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return np.array(json.loads(row["belief_vector"]))
        return None # Retourne None si pas d'état (nouvelle session)