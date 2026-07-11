import os
import datetime
import sqlite3
import requests
import feedparser
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, List, Optional, Annotated
import operator

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from ddgs import DDGS

load_dotenv(override=True)

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# LLMs & Embeddings
fast_model = ChatOpenAI(
    model="nvidia/nemotron-nano-12b-v2-vl:free",
    #openai/gpt-oss-20b:free
    #nvidia/nemotron-nano-12b-v2-vl:free
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE,
    temperature=0.2,
    max_tokens=1200,
    max_retries=5
)

reasoning_model = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    #openai/gpt-oss-120b:free 
    #nvidia/nemotron-3-super-120b-a12b:free 
    #nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
    openai_api_key=OPENROUTER_KEY,
    openai_api_base=OPENROUTER_BASE,
    temperature=0.1,
    max_tokens=2500,
    max_retries=5
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

# Internal Docs initialization
INTERNAL_DOCS_DIR = Path("data/internal_docs")
internal_db = None

def init_internal_db():
    global internal_db
    if internal_db is not None:
        return internal_db
    if not os.path.exists("chroma_internal"):
        if INTERNAL_DOCS_DIR.exists():
            loader = PyPDFDirectoryLoader(str(INTERNAL_DOCS_DIR))
            internal_docs = loader.load()
            internal_chunks = []
            for doc in internal_docs:
                chunks = splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["source_file"] = doc.metadata.get("source", "unknown")
                    chunk.metadata["document_type"] = Path(doc.metadata.get("source", "")).stem
                    internal_chunks.append(chunk)
            if internal_chunks:
                internal_db = Chroma.from_documents(
                    documents=internal_chunks,
                    embedding=embedding_model,
                    persist_directory="chroma_internal",
                    collection_name="internal_knowledge"
                )
    else:
        internal_db = Chroma(
            persist_directory="chroma_internal",
            embedding_function=embedding_model,
            collection_name="internal_knowledge"
        )
    return internal_db

# Tools
@tool
def github_trends(query: str):
    """Recherche les dépôts GitHub les plus populaires sur un sujet donné.
    
    Utilisez cette fonction pour identifier les frameworks, bibliothèques ou projets open-source émergents liés à une technologie.
    
    Arguments :
    query (str) : Les mots-clés de recherche (ex: 'LLM agents', 'LangGraph').
    
    Retour :
    str : Une liste textuelle des 5 dépôts les plus étoilés avec leur description, ou un message d'erreur.
    """
    url = "https://api.github.com/search/repositories"
    params = {"q": query, "sort": "stars", "per_page": 5}
    try:
        r = requests.get(url, params=params)
        data = r.json()
        if "items" not in data:
            return f"Erreur API GitHub ou aucun résultat : {data.get('message', 'Inconnu')}"
        outputs = []
        for repo in data["items"][:5]:
            outputs.append(
                f"Repository: {repo.get('full_name', 'N/A')}\n"
                f"Description: {repo.get('description', 'N/A')}\n"
                f"Stars: {repo.get('stargazers_count', 'N/A')}\n"
            )
        return "\n".join(outputs)
    except Exception as e:
        return f"Erreur lors de la connexion à GitHub : {str(e)}"

@tool
def arxiv_search(query: str):
    """Recherche des articles de recherche académique récents sur arXiv.
    
    Utilisez cette fonction pour trouver les dernières avancées scientifiques, algorithmes ou publications liés à l'IA.
    
    Arguments :
    query (str) : Le sujet de recherche scientifique (ex: 'retrieval augmented generation').
    
    Retour :
    str : Le texte brut des métadonnées et résumés (abstracts) des articles les plus pertinents.
    """
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
    response = requests.get(url)
    return response.text[:4000]

@tool
def rss_ai_news(dummy: str):
    """Lecteur de flux RSS pour l'actualité de l'Intelligence Artificielle.
    
    Utilisez cette fonction pour récupérer les toutes dernières actualités générales, annonces ou failles de sécurité.
    
    Arguments :
    dummy (str) : Argument ignoré, vous pouvez passer une chaîne vide ''.
    
    Retour :
    str : Une liste textuelle contenant les titres et résumés des articles d'actualité récents.
    """
    feeds = [
        "https://feeds.feedburner.com/TheHackersNews",
        "https://www.artificialintelligence-news.com/feed/",
        "https://venturebeat.com/category/ai/feed/"
    ]
    collected = []
    for feed_url in feeds:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:3]:
            collected.append(f"Title: {entry.title}\nSummary: {entry.get('summary', '')}\n")
    return "\n".join(collected)

@tool
def hackernews_ai(dummy: str):
    """Recherche les discussions tendances sur HackerNews concernant les agents IA.
    
    Utilisez cette fonction pour capter les signaux faibles et les opinions de la communauté tech.
    
    Arguments :
    dummy (str) : Argument ignoré, vous pouvez passer une chaîne vide ''.
    
    Retour :
    str : Une liste textuelle contenant les titres et URLs des discussions les plus pertinentes.
    """
    url = "https://hn.algolia.com/api/v1/search"
    params = {"query": "AI agents", "tags": "story", "hitsPerPage": 5}
    r = requests.get(url, params=params)
    data = r.json()
    outputs = [f"Title: {hit.get('title', '')}\nURL: {hit.get('url', '')}\n" for hit in data.get("hits", [])]
    return "\n".join(outputs)

@tool
def tech_news(query: str):
    """Moteur de recherche web généraliste pour l'actualité technologique.
    
    Utilisez cette fonction pour chercher des informations générales ou des événements récents sur le web.
    
    Arguments :
    query (str) : La requête de recherche web (ex: 'régulation IA 2026').
    
    Retour :
    str : Une liste textuelle contenant les titres et extraits des premiers résultats web.
    """
    outputs = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)
        for r in results:
            outputs.append(f"Title: {r.get('title', '')}\nBody: {r.get('body', '')}\n")
    return "\n".join(outputs)

# External DB
external_db = None
def build_external_knowledge(raw_text):
    global external_db
    docs = [
        Document(
            page_content=raw_text,
            metadata={"source": "external_realtime", "date": str(datetime.datetime.now())}
        )
    ]
    chunks = splitter.split_documents(docs)
    external_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_external",
        collection_name="external_knowledge"
    )
    return external_db

# Agents
SCOUT_PROMPT = """Tu es Agent Scout Web.
MISSION : Surveiller GitHub, arXiv, RSS, HackerNews, news IA.
Détecter : nouveautés, signaux faibles, innovations, frameworks émergents.
IMPORTANT : Retourner seulement tendances pertinentes, nouveautés importantes, impacts potentiels."""
scout_agent = create_agent(
    model=fast_model,
    tools=[github_trends, arxiv_search, rss_ai_news, hackernews_ai, tech_news],
    system_prompt=SCOUT_PROMPT
)

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es Agent Analyste Interne. Tu analyses UNIQUEMENT les documents internes de l'entreprise.
    OBJECTIFS : identifier roadmap, budget, technologies, contraintes, projets, risques, gouvernance.
    RÈGLES STRICTES : ne jamais inventer, ne jamais extrapoler, ne jamais utiliser de connaissance externe.
    Si une information n'existe pas, répondre : "Information non disponible". Toutes les affirmations doivent provenir du contexte."""),
    ("human", "CONTEXTE :\n{context}\n\nQUESTION :\n{question}")
])
analyst_chain = ANALYST_PROMPT | fast_model

def build_internal_context(query):
    db = init_internal_db()
    if not db:
        return "Aucune documentation interne disponible."
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20, "lambda_mult": 0.7})
    strategic_query = "roadmap IA, budget IA, stack technique, hallucinations, RAG, LangGraph, AutoGen, projets IA, gouvernance, infrastructure"
    docs = retriever.invoke(strategic_query)
    contexts = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        contexts.append(f"SOURCE: {source}\nCONTENU: {d.page_content}\n")
    return "\n".join(contexts)

COMPARATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es Agent Comparateur.
    Compare les tendances externes et les connaissances internes.
    Détecte : écarts, opportunités, risques, retards, innovations prioritaires.
    IMPORTANT : Produire une vraie analyse stratégique."""),
    ("human", "EXTERNE :\n{external}\n\nINTERNE :\n{internal}")
])
comparison_chain = COMPARATOR_PROMPT | reasoning_model

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es Agent Rédacteur. Génère un rapport professionnel.
    FORMAT :
    # RAPPORT STRATEGIQUE IA
    ## Executive Summary
    ## Tendances clés
    ## Analyse comparative
    ## Opportunités
    ## Risques
    ## Recommandations
    ## Conclusion

    RÈGLES STRICTES : Toute valeur numérique DOIT figurer dans les sources. Ne jamais inventer (ni pourcentage, KPI, budget, SLA, délais).
    Si les données n'existent pas, indiquer "Information non disponible"."""),
    ("human", """ANALYSE A SYNTHETISER :\n{analysis}\n\nFEEDBACK DE L'AGENT VALIDATEUR (si c'est une correction) :\n{feedback}\n
    Si un feedback indique des "ERREURS DETECTEES", corrige impérativement ton précédent rapport en tenant compte de ces remarques.""")
])
writer_chain = WRITER_PROMPT | reasoning_model

VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es Agent Validateur.
    Vérifie : cohérence, absence hallucinations, pertinence, présence sources, logique stratégique.
    Si problème : expliquer précisément.
    Si valide : répondre VALID."""),
    ("human", "RAPPORT :\n{report}")
])
validator_chain = VALIDATOR_PROMPT | reasoning_model

COORDINATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Tu es Agent Coordinateur.
    Tu supervises workflow, qualité, cohérence, orchestration. Tu décides de la prochaine étape, retry, validation finale.
    Tu fonctionnes en logique ReAct : reasoning + acting."""),
    ("human", "ETAT ACTUEL :\n{state}")
])
coordinator_chain = COORDINATOR_PROMPT | reasoning_model

# Graph State
class VeilleState(TypedDict):
    topic: str # Le sujet initial demandé par l'utilisateur (ex: "Roadmap IA")
    external_context: str #Le texte brut compilé par le Scout depuis le web.
    internal_context: str #Les extraits de PDF récupérés par l'Analyste.
    comparison_analysis: str #Le texte généré par le Comparateur (fusion des deux contextes)
    final_report: str #Le rapport Markdown généré par le Rédacteur
    retry_count: int #Nombre de tentatives de coordination (pour éviter les boucles infinies)
    logs: Annotated[List[str], operator.add] #Une liste de logs d'exécution pour le suivi en temps réel (ex: "Scout terminé",..)
    validation: str #Le statut renvoyé par le Validateur (STATUS: PASS ou STATUS: FAIL)
    error: Optional[str] #Pour capturer les erreurs d'API ou autres problèmes critiques.

def init_node(state):
    return {"logs": ["Cycle démarré"], "retry_count": 0}

def coordinator_node(state):
    state_summary = f"Sujet : {state['topic']}\nEssais (retries) restants : {2 - state.get('retry_count', 0)}\nStatut Validation : {state.get('validation', 'En attente')}"
    result = coordinator_chain.invoke({"state": state_summary})
    decision = result.content
    validation = state.get("validation", "")

    if "VALID" in validation:
        return {"logs": ["Coordination (FIN)"]}
    
    retry = state.get("retry_count", 0)
    if retry >= 2:
        return {"logs": ["Validation échouée - abandon"]}

    return {"retry_count": retry + 1, "logs": ["Coordination (ACTION)"]}

def scout_node(state):
    result = scout_agent.invoke({"messages": [HumanMessage(state["topic"])]})
    output = result["messages"][-1].content
    build_external_knowledge(output)
    return {"external_context": output, "logs": ["Scout terminé"]}

def analyst_node(state):
    query = state["topic"]
    internal_context = build_internal_context(query)
    result = analyst_chain.invoke({"context": internal_context, "question": query})
    return {"internal_context": result.content, "logs": ["Analyse interne terminée"]}

def comparator_node(state):
    query = state["topic"]
    global external_db
    rag_externe_content = "Aucune donnée externe en base."
    if external_db is not None:
        external_retriever = external_db.as_retriever(search_kwargs={"k": 5})
        docs = external_retriever.invoke(query)
        rag_externe_content = "\n".join([d.page_content for d in docs])
    combined_external = f"--- DOCS RAG EXTERNE ---\n{rag_externe_content}\n--- NEWS TEMPS REEL ---\n{state['external_context']}"
    result = comparison_chain.invoke({"external": combined_external, "internal": state.get("internal_context", "")})
    return {"comparison_analysis": result.content, "logs": ["Comparator terminé (RAG Hybride)"]}

def writer_node(state):
    feedback = state.get("validation", "Aucun")
    result = writer_chain.invoke({"analysis": state.get("comparison_analysis", ""), "feedback": feedback})
    return {"final_report": result.content, "logs": ["Writer terminé"]}

def validator_node(state):
    report = state.get("final_report", "")
    analysis = state.get("comparison_analysis", "")
    validation_prompt = f"""Tu es un validateur anti-hallucination. Vérifie les faits du rapport.
SOURCES INTERNES:
{state.get('internal_context', '')}
SOURCES EXTERNES:
{state.get('external_context', '')}
ANALYSE COMPARATIVE (utilisée par le rédacteur):
{analysis}

RAPPORT A VALIDER:
{report}

INSTRUCTIONS STRICTES:
Vérifie si les chiffres et faits du rapport proviennent bien des sources ou de l'analyse.
Si le rapport est acceptable, réponds EXACTEMENT: "STATUS: PASS"
S'il y a des hallucinations graves (chiffres inventés), réponds EXACTEMENT: "STATUS: FAIL" suivi de l'explication des erreurs."""
    
    result = fast_model.invoke(validation_prompt)
    return {"validation": result.content, "logs": ["Validation terminée"]}

def validation_router(state):
    validation_result = state.get("validation", "").upper()
    
    # Si OpenRouter renvoie un message d'erreur d'API (ex: Rate Limit 429) sous forme de texte
    if "429" in validation_result or "ERROR" in validation_result or "API" in validation_result:
        print("Erreur d'API OpenRouter détectée. Arrêt du workflow.")
        return END

    if "STATUS: FAIL" in validation_result or "ERREURS" in validation_result[:20]:
        return "writer"
    return END

def coordinator_router(state):
    validation_result = state.get("validation", "").upper()
    if "STATUS: PASS" in validation_result or "VALID" in validation_result or state.get("retry_count", 0) >= 2:
        return "end"
    # Retourner une liste lance les deux agents en parallèle (Fan-out)
    return ["scout", "analyst"]

builder = StateGraph(VeilleState)
builder.add_node("init", init_node)
builder.add_node("coordinator", coordinator_node)
builder.add_node("scout", scout_node)
builder.add_node("analyst", analyst_node)
builder.add_node("comparator", comparator_node)
builder.add_node("writer", writer_node)
builder.add_node("validator", validator_node)

builder.set_entry_point("init")
builder.add_edge("init", "coordinator")
# Fan-out : Le coordinateur déclenche le scout et l'analyste successivement
builder.add_conditional_edges("coordinator", coordinator_router, {"end": END, "scout": "scout", "analyst": "analyst"})
# Fan-in : Le comparateur attend que les deux soient terminés
builder.add_edge("scout", "comparator")
builder.add_edge("analyst", "comparator")
builder.add_edge("comparator", "writer")
builder.add_edge("writer", "validator")
builder.add_conditional_edges("validator", validation_router, {"writer": "writer", END: END})

workflow = builder.compile()

def save_to_history(topic: str, report: str):
    conn = sqlite3.connect("veille_history.db")
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT,
        created_at TEXT,
        report TEXT
    )
    ''')
    cursor.execute(
        "INSERT INTO history (topic, created_at, report) VALUES (?, ?, ?)",
        (topic, datetime.datetime.now().isoformat(), report)
    )
    conn.commit()
    conn.close()

def run_workflow_stream(topic: str):
    """
    Yields events from the workflow.
    """
    initial_state = {"topic": topic, "logs": [], "validation": "", "retry_count": 0}
    config = {"configurable": {"thread_id": "veille_app"}}
    for output in workflow.stream(initial_state, config=config):
        yield output
