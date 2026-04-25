import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

# Importation des fonctions depuis main.py
from main import extract_plate_structured, build_plate_display

# Définition des chemins possibles pour le modèle
MODEL_PATHS = [
    "best.pt",
    "runs/detect/train_extended/weights/best.pt",
    "runs/detect/train/weights/best.pt"
]

@st.cache_resource
def load_model():
    """Charge le modèle YOLO une seule fois et le met en cache."""
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return YOLO(path)
    return None

def main():
    st.set_page_config(
        page_title="ANPR Maroc", 
        page_icon="🇲🇦", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # CSS avancé pour un design premium
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .stApp {
            background: #0a0a0f;
        }

        /* Hero header */
        .hero {
            text-align: center;
            padding: 2rem 0 1rem;
        }
        .hero-badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            color: #818cf8;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            border: 1px solid rgba(99, 102, 241, 0.3);
            margin-bottom: 12px;
        }
        .hero-title {
            font-family: 'Inter', sans-serif;
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #e0e7ff 0%, #818cf8 50%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 8px 0;
            line-height: 1.2;
        }
        .hero-sub {
            font-family: 'Inter', sans-serif;
            font-size: 1rem;
            color: #64748b;
            font-weight: 400;
        }

        /* Glassmorphism cards */
        .glass-card {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 16px;
        }
        .section-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.75rem;
            font-weight: 600;
            color: #6366f1;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 16px;
        }

        /* Result cards */
        .results-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 12px;
            margin: 20px 0;
        }
        .part-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 22px 16px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .part-card:hover {
            background: rgba(255,255,255,0.07);
            border-color: rgba(255,255,255,0.15);
            transform: translateY(-2px);
        }
        .part-card.letter-card {
            background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
            border-color: rgba(99,102,241,0.3);
        }
        .part-card.letter-card:hover {
            border-color: rgba(99,102,241,0.5);
        }
        .part-label {
            font-family: 'Inter', monospace;
            font-size: 0.7rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        }
        .part-value {
            font-family: 'Inter', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            color: #e2e8f0;
            line-height: 1;
        }
        .part-value-ar {
            font-size: 2.8rem;
            font-weight: 700;
            color: #a5b4fc;
            font-family: 'Tahoma', 'Arial', sans-serif;
            line-height: 1;
        }

        /* Plate visual */
        .plate-visual {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 3px solid #1e293b;
            border-radius: 10px;
            padding: 16px 32px;
            text-align: center;
            margin: 16px auto;
            max-width: 420px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.8);
        }
        .plate-visual-text {
            font-family: 'Inter', sans-serif;
            font-size: 2rem;
            font-weight: 900;
            color: #0f172a;
            letter-spacing: 4px;
            margin: 0;
        }
        .plate-flag {
            font-size: 0.65rem;
            color: #64748b;
            font-family: 'Inter', sans-serif;
            margin-top: 4px;
        }

        /* CSV format preview */
        .csv-preview {
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 14px 20px;
            margin-top: 16px;
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
            color: #94a3b8;
            overflow-x: auto;
        }
        .csv-header { color: #6366f1; font-weight: 600; }
        .csv-data { color: #22c55e; }

        /* Upload zone */
        .stFileUploader > div {
            border-radius: 14px !important;
        }
        
        /* Primary button */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s !important;
        }
        .stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
        }

        /* Divider arrow */
        .arrow-sep {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #334155;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero header
    st.markdown("""
        <div class="hero">
            <span class="hero-badge">🇲🇦 Competition ANPR</span>
            <h1 class="hero-title">Reconnaissance de Plaques</h1>
            <p class="hero-sub">Extraction structurée : left_digits · letter_ar · right_digits</p>
        </div>
    """, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("❌ Modèle `best.pt` introuvable à la racine du projet.")
        return

    # Layout en 2 colonnes
    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown('<div class="section-label">📸 Import</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Déposez une image de plaque ici", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            st.image(image, use_container_width=True)
            analyze = st.button("⚡ Analyser la plaque", use_container_width=True, type="primary")

    if uploaded_file is not None and analyze:
        with col_right:
            st.markdown('<div class="section-label">🔍 Résultats</div>', unsafe_allow_html=True)

            with st.spinner("Détection en cours..."):
                results = model(image, verbose=False, conf=0.25)
                preds = results[0].boxes.data.tolist()

                if len(preds) > 0:
                    # Extraction directe depuis les boîtes YOLO
                    left, letter, right = extract_plate_structured(preds, model)
                    display = build_plate_display(left, letter, right)

                    # Grille des 3 parties séparées
                    st.markdown(f"""
                        <div class="results-grid">
                            <div class="part-card">
                                <div class="part-label">left_digits</div>
                                <div class="part-value">{left if left else "—"}</div>
                            </div>
                            <div class="part-card letter-card">
                                <div class="part-label">letter_ar</div>
                                <div class="part-value-ar">{letter if letter else "—"}</div>
                            </div>
                            <div class="part-card">
                                <div class="part-label">right_digits</div>
                                <div class="part-value">{right if right else "—"}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Simulation visuelle d'une plaque
                    st.markdown(f"""
                        <div class="plate-visual">
                            <p class="plate-visual-text" dir="ltr">{left} {letter} {right}</p>
                            <div class="plate-flag">🇲🇦 المغرب</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Preview CSV
                    st.markdown(f"""
                        <div class="csv-preview">
                            <span class="csv-header">image_name,left_digits,letter_ar,right_digits</span><br>
                            <span class="csv-data">{uploaded_file.name},{left},{letter},{right}</span>
                        </div>
                    """, unsafe_allow_html=True)

                    # Image annotée
                    res_bgr = results[0].plot()
                    res_rgb = res_bgr[..., ::-1]
                    st.image(res_rgb, use_container_width=True, caption="Détections YOLO")
                    
                else:
                    st.warning("Aucun caractère détecté sur cette image.")

if __name__ == "__main__":
    main()
