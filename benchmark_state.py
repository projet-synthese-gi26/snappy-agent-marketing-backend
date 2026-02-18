import os
import warnings
# On ignore les warnings futurs de Google pour nettoyer la console
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from convokit import Corpus, download

# Import de TES modules
from app.core.llm import LLamaService
from app.core.belief_llm import BeliefTracker
from app.models.product import ProductProfile

# C'est le contexte exact du dataset PersuasionForGood
P4G_PROFILE = ProductProfile(
    name="Save the Children",
    category="Charity",
    description="Donation for children in need.",
    target_audience="General public",
    price="Donation",
    pain_points=["Hunger", "Poverty", "War", "Lack of education", "Children suffering"],
    benefits=["Feel good", "Tax deduction", "Help humanity"],
    unique_mechanism="Direct aid"
)
# --- 1. CONFIGURATION & MAPPING ---

# Mapping des labels officiels de P4G vers tes États POMDP
# Note : P4G a des labels pour le Persuader (ER) et le Persuadee (EE)
LABEL_MAPPING = {
    # --- Phase d'Introduction (UNAWARE) ---
    "greeting": "UNAWARE",
    "inquire-wellbeing": "UNAWARE",
    "neutral-response": "UNAWARE",
    "ask-not-donate": "UNAWARE", # "Je ne veux pas donner pour l'instant"
    "closing": "CHURNED",        # Fin de discussion

    # --- Phase Problème (PROBLEM_AWARE) ---
    "logical-appeal": "PROBLEM_AWARE",   # Faits, statistiques
    "emotional-appeal": "PROBLEM_AWARE", # Histoires tristes, empathie
    "personal-story": "PROBLEM_AWARE",   # Raconte une histoire
    "inquire-poverty": "PROBLEM_AWARE",  # Pose des questions sur la cause
    "positive-reaction": "PROBLEM_AWARE", # "Oh c'est terrible..."

    # --- Phase Opportunité / Solution (SOLUTION_AWARE) ---
    "credibility-appeal": "SOLUTION_AWARE", # "Nous sommes une ONG reconnue..."
    "ask-org-info": "SOLUTION_AWARE",       # "C'est quoi votre asso ?"
    "self-modeling": "SOLUTION_AWARE",      # "Moi aussi je donne..."

    # --- Phase Produit (PRODUCT_AWARE) ---
    "proposition": "PRODUCT_AWARE",      # "Voulez-vous donner X$ ?"
    "donation-amount": "PRODUCT_AWARE",  # Discute du montant spécifique
    "foot-in-the-door": "PRODUCT_AWARE", # Demande une petite somme

    # --- Phase Action / Conclusion (FULLY_AWARE) ---
    "ask-donation": "FULLY_AWARE",     # Demande directe finale
    "confirm-donation": "FULLY_AWARE", # Confirmation
    "agree-donation": "FULLY_AWARE",   # "Ok je donne"
    
    # --- Rejet (CHURNED / SKEPTICAL) ---
    "disagree-donation": "CHURNED",    # Refus net
    "negative-reaction": "SKEPTICAL",  # Doute
    "source-related-inquiry": "SKEPTICAL" # "Est-ce que l'argent arrive vraiment ?"
}

def load_p4g_data(limit=50):
    print("⏳ Chargement du corpus P4G...")
    try:
        corpus = Corpus(filename=download("persuasionforgood-corpus"))
    except Exception as e:
        print(f"Erreur corpus: {e}")
        return []
        
    data_samples = []
    for utt in corpus.iter_utterances():
        if not utt.text: continue
        
        # Extraction label P4G
        labels_list = utt.meta.get('label_1')
        if not labels_list or not isinstance(labels_list, list): continue
        raw_label = str(labels_list[0]).lower()
        
        if raw_label in LABEL_MAPPING:
            data_samples.append({
                "text": utt.text,
                "pomdp_label": LABEL_MAPPING[raw_label]
            })
        
        if limit and len(data_samples) >= limit: break
            
    print(f"📊 {len(data_samples)} échantillons P4G chargés.")
    return data_samples

def run_benchmark():
    # 1. Chargement Données
    samples = load_p4g_data(limit=30) # Mets 100 ou plus pour le papier
    if not samples: return

    # 2. Init Agent AVEC LE CONTEXTE P4G
    print("🤖 Init Agent avec contexte 'Save the Children'...")
    try:
        llm = LLamaService()
        # ICI : On passe le profil P4G au tracker
        tracker = BeliefTracker(llm, product_profile=P4G_PROFILE)
    except Exception as e:
        print(f"Erreur init: {e}")
        return
    
    y_true = []
    y_pred = []
    
    print(f"\n🚀 Test en cours...")
    for i, item in enumerate(samples):
        text = item['text']
        true_state = item['pomdp_label']
        
        try:
            # L'agent analyse le texte en sachant qu'on parle de charité
            belief = tracker.update_belief(text)
            predicted_state = belief.most_likely_state
            
            y_true.append(true_state)
            y_pred.append(predicted_state)
            
        except Exception:
            pass # On ignore les erreurs API ponctuelles

    # 3. Résultats
    print("\n" + "="*50)
    print("RÉSULTATS SCIENTIFIQUES (Contextualisés P4G)")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    run_benchmark()