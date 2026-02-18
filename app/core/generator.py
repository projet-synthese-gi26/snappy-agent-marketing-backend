from app.core.llm import LLamaService
from app.models.actions import ActionPOPD
from app.models.product import ProductProfile

class ContentGenerator:
    def __init__(self, llm: LLamaService, product: ProductProfile):
        self.llm = llm
        self.product = product

    def generate_text(self, action: ActionPOPD, observation: str) -> str:
        prompt = (
            f"Tu es l'IA de vente pour le produit : {self.product.name}.\n"
            f"PROFIL PRODUIT :\n{self.product.to_prompt_context()}\n\n"
            f"HISTORIQUE :\n{observation}\n\n"
            f"INSTRUCTION STRATÉGIQUE : {action.value}\n"
            "Génère une réponse courte (max 2 phrases), naturelle et persuasive."
        )
        return self.llm.generate(prompt)