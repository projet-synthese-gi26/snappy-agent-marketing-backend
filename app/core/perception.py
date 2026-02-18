import time
from textblob import TextBlob
from typing import Dict, Any

class PerceptionModule:
    def analyze(self, text: str, last_interaction_time: float) -> Dict[str, Any]:
        """Extrait le vecteur O_t = [Latence, Sentiment, Longueur, Intention]"""
        current_time = time.time()
        latency = (current_time - last_interaction_time) if last_interaction_time > 0 else 0
        
        # Sentiment
        try:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
        except:
            sentiment = 0.0
        
        # Longueur
        length = len(text.split())
        
        # Intention (Heuristique rapide pour le POC)
        text_lower = text.lower()
        intent = "AFFIRMATION"
        if "?" in text: intent = "QUESTION"
        elif any(w in text_lower for w in ["non", "pas", "cher", "mais"]): intent = "OBJECTION"
        elif any(w in text_lower for w in ["bonjour", "salut"]): intent = "SALUTATION"
        elif any(w in text_lower for w in ["ok", "d'accord", "je prends"]): intent = "CONFIRMATION"
        elif any(w in text_lower for w in ["prix", "combien", "coût"]): intent = "QUESTION_PRIX"
        
        return {
            "latency": latency,
            "sentiment": sentiment,
            "length": length,
            "intent": intent
        }