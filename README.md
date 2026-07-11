<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-LangGraph-orange.svg" alt="LangGraph">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Vector_DB-ChromaDB-green.svg" alt="ChromaDB">
</div>

<h1 align="center">Cellule SMA : Système Multi-Agents pour la Veille Stratégique</h1>

<p align="center">
  <strong>Automatisation intelligente de la veille stratégique d'entreprise grâce à LangGraph, un RAG Hybride et une boucle stricte anti-hallucination.</strong>
</p>

---

## 📖 Présentation du Projet

La **Cellule SMA** est une application basée sur l'Intelligence Artificielle Générative conçue pour résoudre le problème de la surcharge et de la fragmentation de l'information dans les entreprises. 

Au lieu d'utiliser un simple Chatbot (qui souffre de lacunes contextuelles et d'hallucinations), ce projet déploie un écosystème **Multi-Agents**. Il orchestre plusieurs agents IA spécialisés qui travaillent en parallèle pour croiser des documents confidentiels internes (PDFs) avec des données récupérées en temps réel sur le web (APIs).

## ✨ Fonctionnalités Principales

- 🤖 **Orchestration Multi-Agents (LangGraph) :** Exécution de workflows complexes avec exécution asynchrone (Fan-out) et points de synchronisation (Fan-in).
- 🧠 **RAG Hybride (Retrieval-Augmented Generation) :** 
  - *Interne :* Vectorisation locale via ChromaDB des documents de l'entreprise.
  - *Externe :* Requêtes API en temps réel (GitHub Trends, arXiv, RSS Tech News).
- 🛡️ **Boucle Anti-Hallucination :** Un agent "Validateur" indépendant relit le rapport et force sa réécriture en cas de faits inventés, garantissant une fiabilité à 100%.
- 📊 **Interface Utilisateur Réactive :** Application web Streamlit avec affichage des logs de réflexion des agents en temps réel.
- 📄 **Export Professionnel :** Génération automatique de rapports d'analyse sous format PDF stylisé via `ReportLab`.

## 🏗️ Architecture du Graphe

L'application repose sur un graphe d'états cyclique conçu avec LangGraph :

1. **Coordinateur :** Aiguille la requête et lance la recherche en parallèle.
2. **Scout (Web) & Analyste (Interne) :** Travaillent simultanément. L'analyste interroge la base ChromaDB via l'algorithme MMR, tandis que le Scout utilise des outils web.
3. **Comparateur :** Fusionne les deux contextes pour identifier les écarts stratégiques.
4. **Rédacteur :** Rédige le rapport exécutif en Markdown.
5. **Validateur (Boucle) :** Vérifie le rapport. Si une hallucination est détectée, le graphe effectue un retour en arrière vers le Rédacteur.

*(Consultez le fichier `langgraph_architecture.jpg` à la racine pour un schéma détaillé).*

## 🚀 Installation et Démarrage

### Prérequis
- Python 3.10 ou supérieur.
- Une clé API OpenRouter (pour l'accès gratuit aux modèles open-source).

### Installation

1. **Cloner le dépôt et entrer dans le dossier :**
   ```bash
   git clone <votre_lien_github>
   cd "Cellule SMA"
   ```

2. **Créer un environnement virtuel et installer les dépendances :**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configurer les variables d'environnement :**
   Créez un fichier `.env` à la racine du projet et ajoutez votre clé API :
   ```ini
   OPENROUTER_API_KEY=sk-or-v1-votre-cle-api-ici
   ```

4. **Préparer la base de connaissances interne :**
   Placez vos fichiers PDF d'entreprise (roadmaps, mémos, bilans) dans le dossier `data/internal_docs/`. L'application les vectorisera automatiquement au premier lancement.

### Lancement

Exécutez l'application via Streamlit :
```bash
streamlit run app.py
```
L'interface web s'ouvrira automatiquement dans votre navigateur.

## 📂 Structure du Projet

```text
📁 Cellule SMA/
│
├── app.py                  # Frontend Streamlit (UI et interactions)
├── pipeline.py             # Backend (Logique LangGraph, Agents, Tools, RAG)
├── pdf_export.py           # Module de conversion Markdown -> PDF (ReportLab)
├── requirements.txt        # Liste des dépendances Python
├── .env                    # Fichier de variables d'environnement (API keys)
│
├── 📁 data/
│   └── 📁 internal_docs/   # Placez vos PDFs d'entreprise ici
│
├── 📁 chroma_internal/     # Base de données vectorielle générée localement
└── 📁 reports/             # Dossier où sont sauvegardés les rapports PDF générés
```

## 🛠️ Technologies Utilisées

- **Frameworks IA :** [LangChain](https://python.langchain.com/) / [LangGraph](https://python.langchain.com/docs/langgraph/)
- **Modèles de Langage :** Modèles Open-Source (20B et 120B) via [OpenRouter](https://openrouter.ai/)
- **Base Vectorielle :** [ChromaDB](https://www.trychroma.com/)
- **Embeddings :** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Interface Web :** [Streamlit](https://streamlit.io/)
- **Génération PDF :** ReportLab

