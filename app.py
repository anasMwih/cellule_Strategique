import streamlit as st
import os
import datetime
from pipeline import run_workflow_stream, save_to_history
from pdf_export import generate_pdf

st.set_page_config(page_title="Cellule SMA - Veille Stratégique", page_icon="🤖", layout="wide")

# Custom CSS pour donner un aspect premium
st.markdown("""
<style>
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3498db;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agent de Veille Stratégique (Cellule SMA)")
st.markdown("Bienvenue dans notre outil de veille IA 100% automatisé propulsé par **LangGraph** et de multiples agents.")

with st.sidebar:
    st.header("⚙️ Configuration")
    target_topic = st.text_area(
        "Sujets de veille",
        value="roadmap IA,\nbudget IA,\nstack technique,\nhallucinations,\nRAG,\nLangGraph,\nAutoGen,\nprojets IA,\ngouvernance,\ninfrastructure",
        height=200
    )
    run_btn = st.button("🚀 Lancer l'Analyse", use_container_width=True)

if run_btn:
    if not target_topic.strip():
        st.error("Veuillez entrer au moins un sujet.")
    else:
        st.info("Initialisation des bases de données et des agents...")
        
        # Interface layout
        progress_col, result_col = st.columns([1, 2])
        
        with progress_col:
            st.subheader("📡 Exécution du Workflow")
            status_box = st.empty()
            log_container = st.container()
            
        with result_col:
            st.subheader("📑 Rapport Final")
            report_box = st.empty()

        with st.spinner("Analyse approfondie en cours... (Cela peut prendre quelques minutes)"):
            final_report_content = ""
            
            # Streaming des événements du graphe
            for event in run_workflow_stream(target_topic):
                for node_name, state_update in event.items():
                    
                    # Mise à jour de l'UI
                    status_box.info(f"Étape en cours : **{node_name.upper()}**")
                    
                    # Affichage des logs silencieux
                    logs = state_update.get("logs", [])
                    for log in logs:
                        with log_container:
                            st.write(f"✅ {log}")
                            
                    # Rafraichissement du rapport si disponible
                    if "final_report" in state_update:
                        final_report_content = state_update["final_report"]
                        report_box.markdown(final_report_content)
                        
                    # Gestion visuelle des erreurs d'hallucinations
                    if "validation" in state_update and "ERREURS" in state_update["validation"]:
                        with log_container:
                            st.error("❌ Hallucination détectée ! Le Validateur a renvoyé le rapport au Rédacteur pour correction.")

            status_box.success("🎉 Workflow terminé avec succès !")
            
            # Sauvegardes
            save_to_history(target_topic, final_report_content)
            
            os.makedirs("reports", exist_ok=True)
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
            report_md_path = f"reports/report_{timestamp}.md"
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(final_report_content)
                
            # Génération du PDF
            pdf_path = report_md_path.replace(".md", ".pdf")
            try:
                generate_pdf(final_report_content, pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Télécharger le rapport final (PDF)",
                        data=pdf_file,
                        file_name=f"Rapport_Strategique_IA_{timestamp}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Erreur lors de la génération du PDF: {e}")
