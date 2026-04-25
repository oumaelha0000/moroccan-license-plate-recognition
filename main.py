import os
import random
import shutil
import yaml
import re
import urllib.request
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# CONFIGURATION ET CONSTANTES
# ============================================================
# Nouveaux caractères arabes à ajouter
NEW_CHARS = {
    'yah': 'ﺉ',
    'j':   'ج',
    'ain': 'ع',
    'p':   'ش',
    'ya':  'ي',
}

# 16 classes originales
ORIGINAL_CLASSES = ['0','1','2','3','4','5','6','7','8','9',
                    'a','b','d','h','w','waw']

NEW_CLASS_NAMES = list(NEW_CHARS.keys())
ALL_CLASSES = ORIGINAL_CLASSES + NEW_CLASS_NAMES

# Chemins (à ajuster si vous n'êtes pas sur Colab)
ARABIC_FONT_PATH = "/content/Amiri-Regular.ttf"
SYNTHETIC_DIR    = "/content/synthetic_chars"
ORIGINAL_DATASET = "/content/dataset_maroc"
EXTENDED_DATASET = "/content/dataset_extended"
TEST_FOLDER      = "/content/sample_data/aymane"

# Chemins des modèles
INITIAL_MODEL_PATH = 'runs/detect/train/weights/best.pt'
TRAINED_MODEL_PATH = 'runs/detect/train_extended/weights/best.pt'

# Dictionnaire pour la prédiction
DICT_ARABE = {
    'a': 'أ', 'b': 'ب', 'd': 'د',
    'h': 'ه', 'w': 'و', 'waw': 'و',
}

# Seuils de confiance pour la prédiction
CONF_DIGIT_MIN  = 0.40
CONF_LETTER_MIN = 0.25  # Faible pour lire même avec une faible confiance


# ============================================================
# PARTIE 1 : PRÉPARATION DES DONNÉES
# ============================================================
def setup_environment():
    """Télécharger la police si elle n'existe pas."""
    if not os.path.exists(ARABIC_FONT_PATH):
        print("⏳ Téléchargement de la police Amiri...")
        try:
            urllib.request.urlretrieve(
                "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf", 
                ARABIC_FONT_PATH
            )
            print("✅ Police téléchargée!")
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement de la police: {e}")
    else:
        print("✅ Police déjà présente.")

def create_directories():
    """Créer les dossiers nécessaires pour le dataset étendu."""
    for split in ['train', 'val']:
        os.makedirs(f"{EXTENDED_DATASET}/{split}/images", exist_ok=True)
        os.makedirs(f"{EXTENDED_DATASET}/{split}/labels", exist_ok=True)

def copy_original_dataset():
    """Copier le dataset original vers le nouveau dossier."""
    copied = 0
    for split, split_out in [('train','train'), ('valid','val')]:
        img_dir = f"{ORIGINAL_DATASET}/{split}/images"
        lbl_dir = f"{ORIGINAL_DATASET}/{split}/labels"
        
        if not os.path.exists(img_dir):
            print(f"❌ Dossier introuvable: {img_dir}")
            continue
            
        for f in os.listdir(img_dir):
            shutil.copy(f"{img_dir}/{f}", f"{EXTENDED_DATASET}/{split_out}/images/{f}")
        for f in os.listdir(lbl_dir):
            shutil.copy(f"{lbl_dir}/{f}", f"{EXTENDED_DATASET}/{split_out}/labels/{f}")
            
        copied += len(os.listdir(img_dir))
    print(f"✅ Dataset original copié: {copied} images")

def generate_char_image(char, size=64):
    """Générer des images synthétiques pour un caractère donné."""
    variations = []
    font_sizes = [36, 40, 44, 48]

    for font_size in font_sizes:
        for bg_val in [200, 220, 240, 255]:
            for text_val in [0, 20, 40]:
                img = Image.new('RGB', (size, size), color=(bg_val, bg_val, bg_val))
                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype(ARABIC_FONT_PATH, font_size)
                except:
                    font = ImageFont.load_default()

                bbox = draw.textbbox((0,0), char, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                x = (size - tw) // 2 + random.randint(-4, 4)
                y = (size - th) // 2 + random.randint(-4, 4)

                draw.text((x, y), char, fill=(text_val, text_val, text_val), font=font)

                # Ajouter du bruit
                arr = np.array(img).astype(np.float32)
                arr = np.clip(arr + np.random.normal(0, 10, arr.shape), 0, 255).astype(np.uint8)
                variations.append(Image.fromarray(arr))

    return variations

def create_yolo_label(class_id):
    """Créer le format de label YOLO."""
    return f"{class_id} 0.5 0.5 0.85 0.85\n"

def generate_synthetic_dataset():
    """Générer les images synthétiques pour les nouveaux caractères."""
    print("\n⏳ Génération des caractères synthétiques...")

    for label, char in NEW_CHARS.items():
        class_id = ALL_CLASSES.index(label)
        images = generate_char_image(char)

        # Dupliquer pour avoir 500 images
        all_images = []
        while len(all_images) < 500:
            all_images.extend(images)
        all_images = all_images[:500]
        random.shuffle(all_images)

        split_idx = int(len(all_images) * 0.8)
        splits = {
            'train': all_images[:split_idx],
            'val':   all_images[split_idx:]
        }

        for split, imgs in splits.items():
            for i, img in enumerate(imgs):
                fname = f"synth_{label}_{i:04d}"
                img.save(f"{EXTENDED_DATASET}/{split}/images/{fname}.jpg")
                with open(f"{EXTENDED_DATASET}/{split}/labels/{fname}.txt", 'w') as f:
                    f.write(create_yolo_label(class_id))

        print(f"  ✅ '{label}' ({char}): 500 images (class_id={class_id})")

def create_yaml_config():
    """Créer le fichier data.yaml pour l'entraînement."""
    try:
        n_train = len(os.listdir(f"{EXTENDED_DATASET}/train/images"))
        n_val   = len(os.listdir(f"{EXTENDED_DATASET}/val/images"))

        yaml_content = {
            'path': EXTENDED_DATASET,
            'train': 'train/images',
            'val':   'val/images',
            'nc':    len(ALL_CLASSES),
            'names': ALL_CLASSES
        }

        with open(f"{EXTENDED_DATASET}/data.yaml", 'w') as f:
            yaml.dump(yaml_content, f, allow_unicode=True)

        print(f"\n🏆 Dataset étendu prêt!")
        print(f"   Entraînement: {n_train} images")
        print(f"   Validation:   {n_val} images")
        print(f"   Classes ({len(ALL_CLASSES)}): {ALL_CLASSES}")
    except Exception as e:
        print(f"❌ Erreur lors de la création du fichier YAML: {e}")


# ============================================================
# PARTIE 2 : ENTRAÎNEMENT DU MODÈLE
# ============================================================
def train_model():
    """Entraîner le modèle YOLO."""
    if not os.path.exists(INITIAL_MODEL_PATH):
        print(f"❌ Modèle initial introuvable: {INITIAL_MODEL_PATH}")
        return

    # Important : commencer par les meilleurs poids (transfer learning)
    # Ne pas commencer à zéro, cela permet de conserver les acquis
    print("\n⏳ Début de l'entraînement du modèle...")
    model = YOLO(INITIAL_MODEL_PATH)

    model.train(
        data=f'{EXTENDED_DATASET}/data.yaml',
        epochs=40,
        imgsz=640,
        batch=16,
        patience=15,
        plots=False,
        name='train_extended',
        # Transfer learning — geler le squelette (backbone), entraîner seulement la tête de détection
        freeze=10, 
    )

    print(f"🏆 Terminé! Modèle sauvegardé dans {TRAINED_MODEL_PATH}")


# ============================================================
# PARTIE 3 : PRÉDICTION ET POST-TRAITEMENT
# ============================================================
def get_label(model, class_id):
    return model.names[int(class_id)].lower()

def is_digit(label):
    return label.isdigit()

def is_letter(label):
    return label in DICT_ARABE

def sort_by_x(preds):
    return sorted(preds, key=lambda x: x[0])

def merge_close_boxes(preds, x_threshold=5):
    """Fusionner les boîtes de prédiction trop proches."""
    if not preds:
        return []
    merged = [preds[0]]
    for b in preds[1:]:
        last = merged[-1]
        if abs(b[0] - last[0]) < x_threshold:
            if b[4] > last[4]:
                merged[-1] = b
        else:
            merged.append(b)
    return merged

def detect_layout(preds):
    """
    Détecte si la plaque est sur 1 ligne ou 2 lignes (ex: moto).
    Retourne: 'single' (simple) ou 'double' (double)
    """
    if len(preds) < 3:
        return 'single'

    y_centers = [(b[1] + b[3]) / 2 for b in preds]
    y_sorted = sorted(y_centers)
    y_range = y_sorted[-1] - y_sorted[0]

    if y_range < 8:
        return 'single'

    gaps = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]
    max_gap = max(gaps)

    if max_gap / y_range > 0.25:
        return 'double'
    return 'single'

def split_two_lines(preds):
    """Séparer les boîtes en ligne supérieure et inférieure."""
    y_centers = [(b[1] + b[3]) / 2 for b in preds]
    y_sorted = sorted(y_centers)
    gaps = [y_sorted[i+1] - y_sorted[i] for i in range(len(y_sorted)-1)]
    split_idx = gaps.index(max(gaps))
    threshold = (y_sorted[split_idx] + y_sorted[split_idx+1]) / 2

    top = sort_by_x([b for b in preds if (b[1]+b[3])/2 < threshold])
    bot = sort_by_x([b for b in preds if (b[1]+b[3])/2 >= threshold])
    return top, bot

def boxes_to_string(preds, model):
    """Convertir les boîtes en chaîne de caractères arabes."""
    result = ""
    for b in preds:
        label = get_label(model, b[5])
        conf  = b[4]
        if is_digit(label) and conf >= CONF_DIGIT_MIN:
            result += label
        elif is_letter(label) and conf >= CONF_LETTER_MIN:
            result += DICT_ARABE[label]
    return result

def build_plate(preds, model):
    """
    Logique principale :
    - 1 ligne  → trier par X → lire
    - 2 lignes → détecter haut/bas → combiner les lettres verticalement
    """
    if not preds:
        return ""

    layout = detect_layout(preds)

    if layout == 'single':
        preds = merge_close_boxes(sort_by_x(preds))
        return boxes_to_string(preds, model)

    else:  # double — plaque de moto
        top, bot = split_two_lines(preds)

        # Chiffres = ligne principale (avec le plus grand nombre de chiffres)
        top_str = boxes_to_string(merge_close_boxes(top), model)
        bot_str = boxes_to_string(merge_close_boxes(bot), model)

        # Trouver les chiffres et lettres dans chaque ligne
        top_digits  = re.findall(r'\d+', top_str)
        top_letters = re.findall(r'[^\d\s]+', top_str)
        bot_digits  = re.findall(r'\d+', bot_str)
        bot_letters = re.findall(r'[^\d\s]+', bot_str)

        all_digits  = top_digits + bot_digits
        all_letters = top_letters + bot_letters

        # Combiner les lettres (ex: و + ع = وع)
        combined_letter = ''.join(all_letters)

        # Reconstruire : CHIFFRES_PRINCIPAUX LETTRE(S) CHIFFRES_SUFFIXES
        # Logique : plus de chiffres = numéro principal, moins de chiffres = suffixe
        if len(all_digits) >= 2:
            # Trier les chiffres par longueur décroissante — le plus long = numéro principal
            all_digits_sorted = sorted(all_digits, key=len, reverse=True)
            main_num   = all_digits_sorted[0]
            suffix_num = all_digits_sorted[1] if len(all_digits_sorted) > 1 else ''
        elif len(all_digits) == 1:
            main_num   = all_digits[0]
            suffix_num = ''
        else:
            main_num   = ''
            suffix_num = ''

        if suffix_num:
            return f"{main_num} {combined_letter} {suffix_num}"
        elif main_num:
            return f"{main_num} {combined_letter}"
        else:
            return combined_letter

def format_plate(raw):
    """Nettoyage final de la plaque."""
    raw = raw.strip()
    # S'assurer qu'il y a des espaces autour des lettres arabes
    raw = re.sub(r'(\d+)([^\d\s]+)', r'\1 \2', raw)
    raw = re.sub(r'([^\d\s]+)(\d+)', r'\1 \2', raw)
    return re.sub(r'\s+', ' ', raw).strip()

def extract_plate_parts(formatted_text):
    """
    Sépare le texte formaté en 3 parties: (left_digits, letter_ar, right_digits)
    """
    parts = formatted_text.split()
    left_digits = ""
    letter_ar = ""
    right_digits = ""
    
    for p in parts:
        # Si la partie contient au moins un chiffre
        if re.search(r'\d', p):
            # Tous les chiffres non arabes sont nettoyés de potentiels caractères parasites
            clean_digits = ''.join(filter(str.isdigit, p))
            if not letter_ar:
                left_digits += clean_digits
            else:
                right_digits += clean_digits
        else:
            # Concaténer la lettre
            letter_ar += p

    # Règle de soumission : La lettre arabe doit être fournie comme UN seul caractère
    if len(letter_ar) > 1:
        letter_ar = letter_ar[0]
        
    return left_digits, letter_ar, right_digits

def predict_and_generate_submission():
    """Exécuter la prédiction sur le dossier de test et générer le fichier CSV."""
    if not os.path.exists(TEST_FOLDER):
        print(f"❌ Dossier de test introuvable: {TEST_FOLDER}")
        return

    # Déterminer quel modèle utiliser : le nouveau entraîné ou l'initial
    model_path = TRAINED_MODEL_PATH if os.path.exists(TRAINED_MODEL_PATH) else INITIAL_MODEL_PATH
    if not os.path.exists(model_path):
        print(f"❌ Modèle introuvable pour la prédiction: {model_path}")
        return

    print(f"\n⏳ Chargement du modèle pour la prédiction: {model_path}")
    model_best = YOLO(model_path)

    resultats = []
    images = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    print(f"📸 {len(images)} images à traiter\n")

    for img_name in sorted(images):
        path = os.path.join(TEST_FOLDER, img_name)
        results = model_best(path, verbose=False, conf=0.25)
        preds   = results[0].boxes.data.tolist()

        raw   = build_plate(preds, model_best)
        final = format_plate(raw)

        print(f"✅ {img_name}: '{final}'")
        
        # Extraction selon le nouveau format
        left, letter, right = extract_plate_parts(final)
        
        resultats.append({
            "image_name": img_name, 
            "left_digits": left,
            "letter_ar": letter,
            "right_digits": right
        })

    # Sauvegarde au nouveau format requis pour la compétition
    df = pd.DataFrame(resultats)
    df = df[["image_name", "left_digits", "letter_ar", "right_digits"]]
    
    # Important : utf-8 pur est préféré pour l'évaluation automatique (évite les caractères BOM de utf-8-sig)
    df.to_csv("submission.csv", index=False, encoding='utf-8')
    print(f"\n🏆 submission.csv sauvegardé avec succès au format (left_digits, letter_ar, right_digits) !")
    print(df.head().to_string(index=False)) # Affiche seulement les 5 premières lignes


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================
def main():
    print("=== Démarrage du pipeline ANPR ===")
    
    # 1. Préparation de l'environnement et des données
    setup_environment()
    create_directories()
    copy_original_dataset()
    generate_synthetic_dataset()
    create_yaml_config()
    
    # 2. Entraînement du modèle
    # Décommentez la ligne si vous souhaitez relancer l'entraînement
    # train_model()
    
    # 3. Prédiction sur l'ensemble de test
    predict_and_generate_submission()

if __name__ == "__main__":
    main()
