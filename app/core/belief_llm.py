import json
from app.core.llm import LLamaService
from app.models.states import UserHiddenState, BeliefState
from app.models.product import ProductProfile

class BeliefTracker:
    def __init__(self, llm_service: LLamaService, product_profile :ProductProfile = None):
        self.llm = llm_service
        self.product = product_profile

    def update_belief(self, observation: str) -> BeliefState:
        states_list = [s.name for s in UserHiddenState]
        
        # On injecte le contexte du produit dans le prompt système
        context_str = ""
        if self.product:
            context_str = (
                f"CONTEXTE DU DOMAINE :\n"
                f"Sujet : {self.product.name} ({self.product.category})\n"
                f"Problèmes traités : {', '.join(self.product.pain_points)}\n"
                f"Solution proposée : {self.product.description}\n"
            )

        prompt = (
            f"Tu es un module d'analyse comportementale (POMDP).\n"
            f"{context_str}\n" # <--- Injection du contexte
            f"HISTORIQUE CONVERSATION (Observation) :\n{observation}\n\n"
            f"ÉTATS POSSIBLES : {states_list}\n\n"
            "Tâche : Estime la probabilité entre 0.0 et 1.0 que le client soit dans chacun de ces états.\n"
            "La somme doit faire 1.0.\n"
            "Réponds UNIQUEMENT avec un JSON valide formaté ainsi :\n"
            '{ "UNAWARE": 0.1, "PROBLEM_AWARE": 0.9, ... }'
        )
        
        raw_res = self.llm.generate(prompt)
        clean_json = raw_res.replace("```json", "").replace("```", "").strip()
        
        try:
            probs = json.loads(clean_json)
            # Calcul de l'état dominant
            most_likely = max(probs, key=probs.get)
            return BeliefState(probabilities=probs, most_likely_state=most_likely)
        except Exception:
            # Fallback de sécurité
            return BeliefState(probabilities={"UNAWARE": 1.0}, most_likely_state="UNAWARE")