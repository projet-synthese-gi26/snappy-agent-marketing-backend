import time
import numpy as np
from typing import Tuple, Callable

# Modules
from app.core.llm import LLamaService
from app.core.perception import PerceptionModule
from app.core.belief_bayesian import BayesianBeliefTracker
from app.core.strategy import StrategyAgent
from app.core.generator import ContentGenerator
from app.core.reward import RewardFunction # <--- Intégré
from app.models.product import ProductProfile
from app.models.states import UserHiddenState, BeliefState
from app.models.actions import ActionPOPD

class MarketingAgent:
    def __init__(self, product: ProductProfile):
        print(f"🔧 Init Agent Snappy (Groq) pour {product.name}")
        self.product = product
        self.llm = LLamaService()
        
        # Pipeline
        self.perception = PerceptionModule()
        self.belief_tracker = BayesianBeliefTracker(list(UserHiddenState), list(ActionPOPD))
        self.strategy = StrategyAgent()
        self.generator = ContentGenerator(self.llm, self.product)
        self.reward_func = RewardFunction() # <--- Nouveau
        
        # Mémoire Volatile
        self.last_interaction_time = 0.0
        self.last_action = None
        self.last_belief_state = None # Pour calculer le delta entropie

    def process_turn_for_api(self, history_observation: str, last_user_utterance: str, 
                             belief_manager, session_id: str) -> dict:
        get_belief, update_belief = belief_manager
        
        # 1. Perception
        features = self.perception.analyze(last_user_utterance, self.last_interaction_time)
        self.last_interaction_time = time.time()
        
        # 2. Croyance (POMDP Update)
        prior_vec = get_belief(session_id)
        current_belief = self.belief_tracker.update_belief(
            prior=prior_vec, 
            feats=features, 
            last_action=self.last_action,
            raw_text=last_user_utterance 
        )

        # Sauvegarde pour le tour suivant
        update_belief(session_id, np.array(list(current_belief.probabilities.values())))
        
        # 3. Calcul de la Récompense (Scientific Logging)
        current_reward = 0.0
        if self.last_belief_state and self.last_action:
            current_reward = self.reward_func.compute_reward(
                prev_belief=self.last_belief_state,
                action=self.last_action,
                curr_belief=current_belief
            )
        
        # Mise à jour mémoire agent
        self.last_belief_state = current_belief
        
        # 4. Décision (Policy)
        action = self.strategy.decide_action(current_belief)
        self.last_action = action
        
        # 5. Génération (NLG)
        reply = self.generator.generate_text(action, history_observation)
        
        return {
            "reply": reply, 
            "action": action, 
            "belief": current_belief, 
            "reward": current_reward # On renvoie le reward pour l'analyse
        }

    # Méthode simplifiée pour le benchmark
    def act(self, history_observation: str) -> str:
        last_utt = history_observation.splitlines()[-1] if history_observation.strip() else ""
        feats = self.perception.analyze(last_utt, 0)
        
        # Init simple pour benchmark
        if self.last_belief_state is None:
            self.last_belief_state = BeliefState(
                probabilities=self.belief_tracker.update_belief(self.belief_tracker.get_initial_belief(), feats, None).probabilities,
                most_likely_state="UNAWARE"
            )

        # Update
        prior = np.array(list(self.last_belief_state.probabilities.values()))
        belief = self.belief_tracker.update_belief(prior, feats, self.last_action)
        
        # Reward calculation (optionnel ici mais possible)
        # rew = self.reward_func.compute_reward(self.last_belief_state, self.last_action, belief)
        
        self.last_belief_state = belief
        action = self.strategy.decide_action(belief)
        self.last_action = action
        
        return self.generator.generate_text(action, history_observation)