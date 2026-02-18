from pydantic import BaseModel
from typing import List

class ProductProfile(BaseModel):
    """
    Définit l'ADN du produit. C'est une propriété intrinsèque de l'Agent.
    """
    name: str
    category: str              # ex: "SaaS", "Coaching", "E-commerce"
    description: str           # Pitch court
    target_audience: str       # Qui on vise (Persona)
    price: str                 # ex: "49€/mois"
    
    # Éléments stratégiques pour le POMDP
    pain_points: List[str]     # Liste des douleurs que ça résout (Pour phase PROBLEM_AWARE)
    benefits: List[str]        # Liste des bénéfices (Pour phase SOLUTION_AWARE)
    unique_mechanism: str      # La "sauce secrète" (Pour phase PRODUCT_AWARE)
    
    def to_prompt_context(self) -> str:
        """Serialise le produit pour le contexte du LLM."""
        return (
            f"NOM: {self.name}\n"
            f"CATÉGORIE: {self.category}\n"
            f"CIBLE: {self.target_audience}\n"
            f"PRIX: {self.price}\n"
            f"DOULEURS CIBLÉES: {', '.join(self.pain_points)}\n"
            f"BÉNÉFICES CLÉS: {', '.join(self.benefits)}\n"
            f"MÉCANISME UNIQUE: {self.unique_mechanism}\n"
            f"DESCRIPTION: {self.description}"
        )