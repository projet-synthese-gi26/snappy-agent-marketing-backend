import json, os
from app.models.product import ProductProfile
from dotenv import load_dotenv

def get_env_var(name: str) -> str:
    """
    Récupère une variable d'environnement depuis le système ou un fichier .env.
    Lève une ValueError si la variable est manquante.
    """
    # On essaie d'abord les variables d'environnement système
    value = os.getenv(name)
    if value:
        return value
    
    # Si non trouvé, on charge le .env et on réessaie
    load_dotenv()
    value = os.getenv(name)
    
    if not value:
        raise ValueError(f"Variable d'environnement '{name}' manquante dans le système et dans le fichier .env")
    
    return value

def load_product_config(filepath: str = "product_config.json") -> ProductProfile:
    if not os.path.exists(filepath): raise FileNotFoundError(f"Fichier {filepath} introuvable")
    with open(filepath, 'r', encoding='utf-8') as f:
        return ProductProfile(**json.load(f))