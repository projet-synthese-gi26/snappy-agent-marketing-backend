import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional
from app.models.states import UserHiddenState, BeliefState
from app.models.actions import ActionPOPD

class BayesianBeliefTracker:
    def __init__(self, states: List[UserHiddenState], actions: List[ActionPOPD]):
        self.states = [s.name for s in states]
        self.actions = [a.name for a in actions]
        self.state_map = {name: i for i, name in enumerate(self.states)}
        self.num_states = len(self.states)
        
        # Tenseur de Transition T(s'|s, a)
        self.transition_tensor = {}
        self._init_transitions()
        
        # Matrice d'Observation Z (Définie par densités)
        self.expectations = {
            "UNAWARE": {"sentiment": (0.2, 0.4), "length": (4, 4), "intent": "SALUTATION"},
            "PROBLEM_AWARE": {"sentiment": (-0.4, 0.4), "length": (15, 10), "intent": "AFFIRMATION"},
            "SOLUTION_AWARE": {"sentiment": (0.1, 0.3), "length": (10, 5), "intent": "QUESTION"},
            "PRODUCT_AWARE": {"sentiment": (0.0, 0.3), "length": (8, 4), "intent": "QUESTION_PRIX"},
            "SKEPTICAL": {"sentiment": (-0.3, 0.3), "length": (12, 6), "intent": "OBJECTION"},
            "FULLY_AWARE": {"sentiment": (0.6, 0.3), "length": (5, 3), "intent": "CONFIRMATION"},
            "CHURNED": {"sentiment": (-0.6, 0.4), "length": (4, 3), "intent": "OBJECTION"}
        }

    def _init_transitions(self):
        # Matrice d'identité lissée (Transition par défaut)
        base = np.eye(self.num_states) + 0.05
        base /= base.sum(axis=1, keepdims=True)
        s_map = self.state_map

        for action in self.actions:
            T = base.copy()
            # Dynamique causale de l'article (Eq. 2)
            if action == "QUESTIONNER_DOULEUR":
                T[s_map["UNAWARE"], s_map["PROBLEM_AWARE"]] += 2.0
            elif action == "AMPLIFIER_PROBLEME":
                T[s_map["PROBLEM_AWARE"], s_map["SOLUTION_AWARE"]] += 1.5
            elif action == "VIRAGE_OPPORTUNITE":
                T[s_map["PROBLEM_AWARE"], s_map["SOLUTION_AWARE"]] += 2.0
            elif action == "PITCH_SOLUTION":
                T[s_map["SOLUTION_AWARE"], s_map["PRODUCT_AWARE"]] += 2.5
            elif action == "APPEL_ACTION_HARD":
                T[s_map["PRODUCT_AWARE"], s_map["FULLY_AWARE"]] += 3.0
                T[s_map["PRODUCT_AWARE"], s_map["CHURNED"]] += 1.0 # Risque
            
            # Normalisation
            self.transition_tensor[action] = T / T.sum(axis=1, keepdims=True)

    def _likelihood(self, feats: Dict, state: str) -> float:
        exp = self.expectations[state]
        p_s = norm.pdf(feats['sentiment'], exp['sentiment'][0], exp['sentiment'][1])
        p_l = norm.pdf(feats['length'], exp['length'][0], exp['length'][1])
        p_i = 0.8 if feats['intent'] == exp['intent'] else 0.1
        return (p_s * p_l * p_i) + 1e-9

    def update_belief(self, prior: np.ndarray, feats: Dict, last_action: Optional[ActionPOPD]) -> BeliefState:
        # 1. Transition (Prédiction)
        T = self.transition_tensor.get(last_action.name if last_action else "ATTENTE", np.eye(self.num_states))
        pred = prior @ T
        
        # 2. Observation (Correction)
        lik = np.array([self._likelihood(feats, s) for s in self.states])
        post = pred * lik
        
        # Normalisation
        if np.sum(post) > 0: post /= np.sum(post)
        else: post = pred
        
        probs = {n: p for n, p in zip(self.states, post)}
        return BeliefState(probabilities=probs, most_likely_state=self.states[np.argmax(post)])

    def get_initial_belief(self):
        p = np.zeros(self.num_states)
        p[self.state_map["UNAWARE"]] = 1.0
        return p