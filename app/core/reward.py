import numpy as np
from app.models.states import BeliefState
from app.models.actions import ActionPOPD

class RewardFunction:
    """
    Implémente l'équation (5) de l'article :
    R = Rout - Cost + Shaping + lambda * DeltaEntropy
    """
    def __init__(self, gamma=0.9, lambda_entropy=0.5):
        self.gamma = gamma
        self.lambda_entropy = lambda_entropy
        
        # Potentiels Φ(s) pour le Reward Shaping (Ng et al., 1999)
        self.state_potentials = {
            "UNAWARE": 0.0,
            "PROBLEM_AWARE": 0.3,
            "SOLUTION_AWARE": 0.5,
            "PRODUCT_AWARE": 0.7,
            "SKEPTICAL": 0.4,
            "FULLY_AWARE": 1.0,  # Le but
            "CHURNED": -1.0      # L'échec
        }

    def _entropy(self, belief: BeliefState) -> float:
        """Calcule l'entropie de Shannon H(B) = -Sum(p * log(p))"""
        H = 0.0
        for p in belief.probabilities.values():
            if p > 1e-6: # Évite log(0)
                H -= p * np.log(p)
        return H

    def _potential(self, belief: BeliefState) -> float:
        """Calcule Φ(B) = Espérance du potentiel des états"""
        phi = 0.0
        for state, prob in belief.probabilities.items():
            phi += prob * self.state_potentials.get(state, 0.0)
        return phi

    def compute_reward(self, 
                       prev_belief: BeliefState, 
                       action: ActionPOPD, 
                       curr_belief: BeliefState) -> float:
        
        # 1. Récompense Immédiate / Terminale (Rout)
        r_out = 0.0
        top_state = curr_belief.most_likely_state
        
        # Si on a réussi à vendre (État final positif)
        if top_state == "FULLY_AWARE" and action == ActionPOPD.APPEL_ACTION_HARD:
            r_out = 50.0
        # Si le client est parti (État final négatif)
        elif top_state == "CHURNED":
            r_out = -20.0
            
        # 2. Coût d'action (C) - Pénalise la longueur
        cost = 1.0

        # 3. Reward Shaping (F = gamma * Phi(t+1) - Phi(t))
        shaping = (self.gamma * self._potential(curr_belief)) - self._potential(prev_belief)

        # 4. Gain d'Information (Réduction d'entropie)
        delta_entropy = self._entropy(prev_belief) - self._entropy(curr_belief)

        # Total
        return r_out - cost + shaping + (self.lambda_entropy * delta_entropy)