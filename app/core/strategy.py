import numpy as np
from app.models.actions import ActionPOPD
from app.models.states import BeliefState

class StrategyAgent:
    def __init__(self, temperature=0.6):
        self.temperature = temperature
        # Q-Table Experte (Simulée)
        self.q_table = {
            "UNAWARE": {"QUESTIONNER_DOULEUR": 10.0, "EMPATHIE": 5.0},
            "PROBLEM_AWARE": {"VIRAGE_OPPORTUNITE": 10.0, "AMPLIFIER_PROBLEME": 7.0},
            "SOLUTION_AWARE": {"PITCH_SOLUTION": 12.0, "PREUVE_SOCIALE": 5.0},
            "PRODUCT_AWARE": {"APPEL_ACTION_SOFT": 8.0, "GESTION_OBJECTION": 6.0},
            "SKEPTICAL": {"GESTION_OBJECTION": 15.0, "PREUVE_SOCIALE": 8.0},
            "FULLY_AWARE": {"APPEL_ACTION_HARD": 20.0},
            "CHURNED": {"ATTENTE": 10.0}
        }

    def decide_action(self, belief: BeliefState) -> ActionPOPD:
        actions = list(ActionPOPD)
        agg_q = {a.name: 0.0 for a in actions}

        # Q(b,a) = Sum(b(s) * Q(s,a))
        for state, prob in belief.probabilities.items():
            if prob < 0.01: continue
            q_s = self.q_table.get(state, {})
            for a in actions:
                agg_q[a.name] += prob * q_s.get(a.name, 0.0)

        # Softmax (Eq. 9)
        q_vec = np.array([agg_q[a.name] for a in actions])
        q_vec -= np.max(q_vec)
        exp_q = np.exp(q_vec / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        return np.random.choice(actions, p=probs)