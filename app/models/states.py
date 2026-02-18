from enum import Enum
from pydantic import BaseModel
from typing import Dict

class UserHiddenState(Enum):
    UNAWARE = "Inconscient du problème"
    PROBLEM_AWARE = "Conscient de la douleur"
    SOLUTION_AWARE = "Cherche une solution"
    PRODUCT_AWARE = "Compare le produit"
    FULLY_AWARE = "Prêt à acheter"
    SKEPTICAL = "Doute / Objection"
    CHURNED = "Désintéressé / Parti"
    CONVERTED = "Achat effectué / Succès confirmé"

class BeliefState(BaseModel):
    probabilities: Dict[str, float]
    most_likely_state: str