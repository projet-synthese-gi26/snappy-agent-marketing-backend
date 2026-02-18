from enum import Enum

class ActionPOPD(Enum):
    QUESTIONNER_DOULEUR = "Poser une question sur le problème"
    AMPLIFIER_PROBLEME = "Souligner les conséquences du problème"
    VIRAGE_OPPORTUNITE = "Suggérer une solution"
    PITCH_SOLUTION = "Présenter le produit"
    PREUVE_SOCIALE = "Donner un exemple de réussite"
    APPEL_ACTION_SOFT = "Proposer une première étape"
    APPEL_ACTION_HARD = "Demander l'achat"
    GESTION_OBJECTION = "Répondre à une objection"
    EMPATHIE = "Montrer de l'empathie"
    ATTENTE = "Écouter / Saluer"
    CLOTURE = "Remercier et fermer la conversation"