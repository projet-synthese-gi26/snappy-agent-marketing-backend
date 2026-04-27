
# Snappy : Agent Conversationnel Marketing Probabiliste (POMDP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688)
![License](https://img.shields.io/badge/License-MIT-green)

**Snappy** est un agent conversationnel de nouvelle génération conçu pour résoudre le problème de l'**incertitude épistémique** dans l'automatisation de la relation client.

Contrairement aux chatbots basés uniquement sur des LLMs (qui hallucinent des stratégies), Snappy utilise une architecture hybride fondée sur un **Processus Décisionnel Markovien Partiellement Observable (POMDP)**. Il maintient une croyance probabiliste sur l'état psychologique du client (*Belief State*) pour prendre des décisions stratégiques optimales avant de générer du texte.

---

## 🧠 Architecture Scientifique

L'agent repose sur le découplage strict entre la **Décision** (Le Cerveau) et la **Génération** (La Bouche).

1.  **Perception ($O$) :** Extraction de signaux quantitatifs (Latence, Sentiment, Intention) via NLP classique.
2.  **State Tracking ($B$) :** Un **Filtre Bayésien** met à jour la distribution de probabilité sur les états cachés du client (ex: `SKEPTICAL`, `READY_TO_BUY`) en utilisant :
    *   Un **Tenseur de Transition** ($T$) issu de la théorie de la vente.
    *   Une **Matrice d'Observation** ($Z$) basée sur des lois Log-Normales.
3.  **Politique ($\pi$) :** Une stratégie **Softmax** sélectionne l'action optimale (ex: `GESTION_OBJECTION`) basée sur des Q-Values.
4.  **Génération :** Un LLM (via **Groq**) transforme l'action abstraite en réponse naturelle, contextualisée par le produit.

---

## ✨ Fonctionnalités Clés

*   **Gestion de l'Incertitude :** Ne fonce pas tête baissée. Si l'agent est incertain (ex: 51% Intéressé / 49% Sceptique), il pose des questions pour lever l'ambiguïté.
*   **Produit Agnostique :** L'agent est configuré dynamiquement via un fichier JSON. Il peut vendre des panneaux solaires, des logiciels ou des chaussures sans changer une ligne de code.
*   **Détection de Clôture :** Identifie formellement la fin de transaction (`CONVERTED`) ou le rejet (`CHURNED`) pour arrêter la boucle.

---

## 🛠️ Installation

### Prérequis
*   Python 3.9+
*   Une clé API **Groq** (pour le LLM rapide Llama-3).

### Exécution

```bash
cd snappy-agent-marketing

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt

# Configuration
# Créer un fichier .env à la racine et ajouter :
# GROQ_API_KEY=votre_cle_api_groq
```

### 🐳 Docker (Optionnel)

Si vous préférez utiliser Docker, un `Dockerfile` est disponible.

**1. Construire l'image :**
```bash
docker build -t snappy-backend .
```

**2. Lancer le conteneur :**
Pensez à passer votre clé API Groq via une variable d'environnement.
```bash
docker run -p 8000:8000 -e GROQ_API_KEY=votre_cle_api_ici snappy-backend
```

---

## 🚀 Démarrage

###  Lancer le Cerveau (API)
```bash
uvicorn app.main:app --reload
```
L'API sera accessible sur `http://localhost:8000`. La documentation Swagger est sur `/docs`.


---

## 📊 Validation Scientifique (Benchmarks)

Le projet inclut des scripts de benchmarking pour valider la thèse de l'article associé.

### Test 1 : Taux de Succès & Robustesse
Compare l'agent POMDP contre une baseline (Vanilla LLM) et un agent aléatoire sur des scénarios de vente simulés.
```bash
python benchmark_success.py
```
*Génère un rapport `benchmark_success_results.csv` comparant le Taux de Conversion et le nombre de tours.*

### Test 2 : Précision du Suivi d'État (State Tracking)
Évalue la capacité du Tracker Bayésien à deviner l'état réel du client sur le dataset standard **PersuasionForGood**.
```bash
python benchmark_state.py
```
*Calcule l'Accuracy et le F1-Score du module de perception.*

---

## ⚙️ Configuration du Produit

Pour changer ce que l'agent vend, modifiez simplement `product_config.json` :

```json
{
  "name": "EcoHome Solar",
  "category": "Énergie Renouvelable",
  "price": "699€",
  "pain_points": ["Factures élevées", "Coupures de courant"],
  "benefits": ["Autonomie", "Économies"],
  "unique_mechanism": "Technologie SunSync"
}
```
L'agent adaptera instantanément sa stratégie et son argumentaire.

---

## 📂 Structure du Projet

```text
/app
├── core/                # Le Cœur Intelligent
│   ├── belief_bayesian.py # Calcul des probabilités (POMDP)
│   ├── strategy.py      # Prise de décision (Policy)
│   ├── perception.py    # Analyse du signal (O)
│   └── reward.py        # Fonction de récompense (R)
├── models/              # Définitions des États et Actions
├── services/            # Gestion de session et config
└── main.py              # Point d'entrée API
```

---

## 👥 Auteurs


*   **École Nationale Supérieure Polytechnique de Yaoundé**
*   *Département de Génie Informatique*

---

> "La confiance ne se décrète pas, elle se construit par la pertinence."