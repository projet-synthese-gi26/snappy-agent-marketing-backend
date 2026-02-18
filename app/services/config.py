import json, os
from app.models.product import ProductProfile

def load_product_config(filepath: str = "product_config.json") -> ProductProfile:
    if not os.path.exists(filepath): raise FileNotFoundError(f"Fichier {filepath} introuvable")
    with open(filepath, 'r', encoding='utf-8') as f:
        return ProductProfile(**json.load(f))