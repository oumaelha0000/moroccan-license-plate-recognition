import streamlit as st
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

# Importation des fonctions de traitement métier depuis main.py
from main import build_plate, format_plate

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
    # Configuration de la page Streamlit
    st.set_page_config(page_title="ANPR Maroc - Détecteur de Plaques", page_icon="🚗", layout="centered")

    st.title("🚗 Reconnaissance de Plaques d'Immatriculation - Maroc")
    st.markdown("Téléchargez l'image d'un véhicule (ou directement sa plaque), et l'intelligence artificielle détectera, lira et formatera le texte exact de la plaque d'immatriculation.")

    # Chargement du modèle
    model = load_model()

    if model is None:
        st.error("❌ Modèle non trouvé ! Assurez-vous que le fichier `best.pt` est bien présent à la racine de votre projet.")
        return

    # Interface d'import d'image
    uploaded_file = st.file_uploader("Prenez une photo ou importez une image (JPG, PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Affichage de l'image originale
        image = Image.open(uploaded_file)
        
        # S'assurer que l'image est bien en RGB (pour éviter les soucis avec les espaces de couleur)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        st.image(image, caption='Image Originale', use_column_width=True)
        
        # Bouton pour lancer l'analyse
        if st.button("🔍 Extraire le numéro de plaque", use_container_width=True):
            with st.spinner("Analyse en cours par l'IA..."):
                
                # YOLO inference
                results = model(image, verbose=False, conf=0.25)
                
                # Extraction des boîtes de prédiction
                preds = results[0].boxes.data.tolist()

                st.markdown("### 📊 Résultats")
                
                if len(preds) > 0:
                    # Traitement NLP/Règles via les fonctions de main.py
                    raw_text = build_plate(preds, model)
                    final_text = format_plate(raw_text)
                    
                    # Rendu visuel
                    st.success("Lecture terminée !")
                    st.markdown(
                        f"""
                        <div style="background-color:#F0F2F6; padding:20px; border-radius:10px; text-align:center; border:2px solid #4CAF50;">
                            <h1 style="color:#2E4053; margin:0px; font-family:'Amiri', sans-serif;" dir="rtl">
                                {final_text}
                            </h1>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Extraction et affichage de l'image avec les Bounding boxes
                    # plot() retourne un BGR numpy array
                    res_image_bgr = results[0].plot()
                    # Convertir en RGB
                    res_image_rgb = res_image_bgr[..., ::-1]
                    st.image(res_image_rgb, caption="Détections du modèle", use_column_width=True)
                    
                else:
                    st.warning("Aucun caractère n'a été détecté avec un niveau de confiance suffisant.")

if __name__ == "__main__":
    main()
