import time
import random
import pandas as pd
from tqdm import tqdm
from typing import Tuple

# --- Imports de ton architecture ---
from app.core.llm import LLamaService
from app.core.agent import MarketingAgent # L'agent POMDP unifié
from app.models.product import ProductProfile
from app.models.actions import ActionPOPD

# --- 1. LE SIMULATEUR D'UTILISATEUR (Adaptatif) ---
class UserSimulator:
    def __init__(self, product: ProductProfile):
        self.llm = LLamaService()
        self.product = product # Le simulateur connaît le produit pour être un client pertinent
        self.history = []
        self.max_turns = 15

    def reset(self) -> str:
        """Réinitialise le client avec un but lié au produit."""
        self.history = []
        # Le but du client est de résoudre un des problèmes que le produit cible
        self.goal = f"Trouver une solution pour '{random.choice(self.product.pain_points)}'"
        self.persona = self.product.target_audience
        self.resistance = random.randint(4, 8)
        return "Bonjour, je cherche des informations."

    def step(self, agent_message: str) -> Tuple[str, bool, bool]:
        """Retourne (Réponse, Fini, Succès)."""
        self.history.append(f"Vendeur: {agent_message}")
        
        prompt = (
            f"Jeu de rôle. Tu es un client potentiel.\n"
            f"PROFIL CLIENT : {self.persona}\n"
            f"OBJECTIF SECRET : {self.goal}\n"
            f"PRODUIT PROPOSÉ : {self.product.name}\n"
            f"HISTORIQUE :\n" + "\n".join(self.history[-6:]) + "\n\n"
            "RÈGLES :\n"
            "- Réponds naturellement en 1-2 phrases.\n"
            "- Si le vendeur te convainc, dis explicitement '[ACHAT]'.\n"
            "- Si tu en as marre, dis '[QUITTER]'.\n"
            "Réponse :"
        )
        
        user_response = self.llm.generate(prompt)
        self.history.append(f"Moi: {user_response}")
        
        done = "[ACHAT]" in user_response or "[QUITTER]" in user_response or len(self.history) >= self.max_turns * 2
        success = "[ACHAT]" in user_response
            
        return user_response, done, success

# --- 2. LES AGENTS DE COMPARAISON (Baselines) ---

class RandomAgent:
    """Baseline Zéro : Agit au hasard."""
    def __init__(self, product: ProductProfile):
        self.actions = list(ActionPOPD)
        self.llm = LLamaService()
        self.product_name = product.name

    def act(self, observation: str) -> str:
        action = random.choice(self.actions)
        return self.llm.generate(f"Dis une phrase aléatoire sur {self.product_name} en suivant l'idée : {action.value}")

class VanillaLLMAgent:
    """Baseline Forte : Un LLM seul, sans POMDP."""
    def __init__(self, product: ProductProfile):
        self.llm = LLamaService()
        self.product = product

    def act(self, observation: str) -> str:
        prompt = (
            f"Tu es un vendeur pour le produit suivant :\n"
            f"{self.product.to_prompt_context()}\n\n"
            f"HISTORIQUE :\n{observation}\n\n"
            "Réponds au client pour le convaincre d'acheter."
        )
        return self.llm.generate(prompt)

# --- 3. DÉFINITION DES PRODUITS DE TEST ---

TEST_PRODUCTS = [
    ProductProfile(
        name="CryptoGuard Wallet",
        category="Cybersécurité",
        description="Clé USB physique pour sécuriser vos cryptomonnaies.",
        target_audience="Investisseurs crypto, débutants ou experts.",
        price="150€",
        pain_points=["Peur du hack de plateformes", "Complexité des wallets logiciels", "Perte de clés privées"],
        benefits=["Stockage hors-ligne impossible à pirater", "Interface ultra-simple", "Backup physique"],
        unique_mechanism="Puce de chiffrement EAL5+ inviolable."
    ),
    ProductProfile(
        name="ZenMat",
        category="Bien-être",
        description="Tapis d'acupression pour la relaxation et le soulagement des douleurs.",
        target_audience="Personnes stressées avec des douleurs de dos.",
        price="29.99€",
        pain_points=["Mal de dos chronique", "Difficulté à s'endormir", "Stress et anxiété"],
        benefits=["Soulagement de la douleur en 15 minutes", "Amélioration de la qualité du sommeil", "Relaxation profonde"],
        unique_mechanism="6,210 pointes de stimulation brevetées en forme de lotus."
    )
]

# --- 4. LE BANC D'ESSAI ---

def run_single_episode(agent, simulator) -> Tuple[bool, int]:
    """Joue une conversation complète et retourne (succès, nb_tours)."""
    observation_history = f"Client: {simulator.reset()}\n"
    turns = 0
    done = False
    
    while not done:
        turns += 1
        bot_msg = agent.act(observation_history)
        observation_history += f"Agent: {bot_msg}\n"
        
        user_msg, done, success = simulator.step(bot_msg)
        observation_history += f"Client: {user_msg}\n"
        
    return success, turns

def run_benchmark(num_episodes_per_product=5):
    print(f"🚀 Démarrage du Benchmark de Taux de Succès...")
    
    results = []

    # Boucle sur les produits à tester
    for product in TEST_PRODUCTS:
        print(f"\n📦 Test du produit : {product.name}")
        
        # On définit les agents à comparer POUR CE PRODUIT
        agents_to_test = {
            "1. Random (Baseline)": RandomAgent(product),
            "2. Vanilla LLM (Baseline)": VanillaLLMAgent(product),
            "3. Ours (POMDP)": MarketingAgent(product) # Utilise ta classe unifiée
        }
        
        # Boucle sur les agents
        for agent_name, agent in agents_to_test.items():
            print(f"  -> Test de l'agent : {agent_name}")
            wins = 0
            total_turns = 0
            
            # Boucle sur les simulations
            for _ in tqdm(range(num_episodes_per_product)):
                sim = UserSimulator(product)
                is_success, n_turns = run_single_episode(agent, sim)
                if is_success: wins += 1
                total_turns += n_turns
            
            # Stockage des résultats
            results.append({
                "Produit": product.name,
                "Modèle": agent_name,
                "Success Rate (%)": (wins / num_episodes_per_product) * 100,
                "Avg Turns": round(total_turns / num_episodes_per_product, 2)
            })

    # Affichage final
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("RÉSULTATS FINAUX (Benchmark de Taux de Succès)")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv("benchmark_success_results.csv", index=False)
    print("\nSauvegardé dans 'benchmark_success_results.csv'.")

if __name__ == "__main__":
    # Attention aux coûts API. 5 épisodes x 2 produits x 3 agents = 30 conversations.
    run_benchmark(num_episodes_per_product=5)