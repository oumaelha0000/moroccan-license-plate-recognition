import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

# Importation des fonctions depuis main.py
from main import build_plate, format_plate, extract_plate_parts

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
    # Configuration globale avec un layout plus large
    st.set_page_config(page_title="ANPR Maroc - Premium", page_icon="🇲🇦", layout="wide")

    # Injection de CSS pour un design dynamique et moderne
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        /* Cacher le menu Streamlit par défaut et le footer si besoin */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .main {
            background-color: #0E1117;
        }
        .title {
            font-family: 'Roboto', sans-serif;
            text-align: center;
            background: -webkit-linear-gradient(45deg, #4CAF50, #81C784);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #A0AEC0;
            margin-bottom: 3rem;
        }
        
        /* Conteneur principal pour la plaque finale combinée */
        .plate-box {
            background: linear-gradient(135deg, #1A202C 0%, #2D3748 100%);
            border: 2px solid #333;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            transition: transform 0.3s ease, border 0.3s ease;
            margin-top: 10px;
            margin-bottom: 20px;
        }
        .plate-box:hover {
            transform: translateY(-5px);
            border-color: #4CAF50;
        }
        .plate-text {
            color: #FFFFFF;
            font-size: 2.5rem;
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            letter-spacing: 2px;
            font-weight: bold;
        }

        /* Conteneur pour les 3 métriques séparées correspondants au format de soumission */
        .result-container {
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 20px 15px;
            flex: 1;
            min-width: 120px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: all 0.3s;
        }
        .result-card:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255,255,255,0.3);
        }
        .card-highlight {
            border: 1px solid rgba(76, 175, 80, 0.5);
            background: rgba(76, 175, 80, 0.1);
        }
        
        .metric-label {
            color: #A0AEC0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }
        .metric-value {
            color: #E2E8F0;
            font-size: 2rem;
            font-weight: 700;
        }
        .metric-value-ar {
            color: #4CAF50;
            font-size: 2.5rem;
            font-weight: bold;
            font-family: Tahoma, Arial, sans-serif;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>Système de Reconnaissance de Plaques</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Identification précise respectant le <b>Nouveau Format de Soumission</b> LTR/RTL</p>", unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.error("❌ Modèle non trouvé ! Assurez-vous que le fichier `best.pt` est bien présent à la racine.")
        return

    # Disposition en deux colonnes pour le dashboard
    col_upload, col_result = st.columns([1, 1.2])

    with col_upload:
        st.markdown("<h3 style='color:#E2E8F0;'>1. Importation d'image</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Affichage de l'image de base
            st.image(image, use_column_width=True, caption="Image source déposée")
            analyze_button = st.button("Lancer l'analyse par l'IA", use_container_width=True, type="primary")

    if uploaded_file is not None and analyze_button:
        with col_result:
            st.markdown("<h3 style='color:#E2E8F0;'>2. Analyse & Extraction</h3>", unsafe_allow_html=True)
            with st.spinner("Vérification en cours..."):
                results = model(image, verbose=False, conf=0.25)
                preds = results[0].boxes.data.tolist()

                if len(preds) > 0:
                    raw_text = build_plate(preds, model)
                    final_text = format_plate(raw_text)
                    
                    # Nouveau format de soumission
                    left, letter, right = extract_plate_parts(final_text)
                    
                    st.success("Détection réussie !")

                    # Présentation des caractéristiques séparées pour la soumission
                    st.markdown(f"""
                        <div class="result-container">
                            <div class="result-card">
                                <div class="metric-label">left_digits</div>
                                <div class="metric-value">{left if left else "-"}</div>
                            </div>
                            <div class="result-card card-highlight">
                                <div class="metric-label">letter_ar</div>
                                <div class="metric-value-ar">{letter if letter else "-"}</div>
                            </div>
                            <div class="result-card">
                                <div class="metric-label">right_digits</div>
                                <div class="metric-value">{right if right else "-"}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Preview global
                    st.markdown(f"""
                    <div class="plate-box">
                        <div class="metric-label" style="margin-bottom:-5px;">Format visuel complet</div>
                        <p class="plate-text" dir="rtl">{final_text}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Bounding boxes visuelles
                    res_image_bgr = results[0].plot()
                    res_image_rgb = res_image_bgr[..., ::-1]
                    
                    st.image(res_image_rgb, use_column_width=True, caption="Vision par ordinateur (Bounding boxes)")
                    st.balloons()
                else:
                    st.error("Aucune plaque détectée avec suffisamment de confiance sur cette image.")

if __name__ == "__main__":
    main()
