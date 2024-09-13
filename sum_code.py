import os
import cv2
import shutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import joblib
import pandas as pd

# Télécharger les ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialiser le processeur et le modèle Blip
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Fonction pour détecter la violence dans une image
def detect_violence(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return False
    else:
        return True

# Fonction pour convertir une vidéo en images (frames)
def video_to_frames(video_path, output_folder_frames):
    if not os.path.exists(output_folder_frames):
        os.makedirs(output_folder_frames)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_path = os.path.join(output_folder_frames, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1
        else:
            break

    cap.release()

    return output_folder_frames

# Fonction pour filtrer les images significatives et générer des légendes
def filter_and_generate_captions(input_folder_frames, output_folder, output_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with open(output_file, "w") as file:
            for filename in os.listdir(input_folder_frames):
                if filename.endswith(".jpg"):
                    frame_path = os.path.join(input_folder_frames, filename)

                    if detect_violence(frame_path):
                        shutil.copy(frame_path, output_folder)

                        raw_image = Image.open(frame_path).convert('RGB')
                        inputs = processor(raw_image, return_tensors="pt")
                        out = model.generate(**inputs)
                        generated_caption = processor.decode(out[0], skip_special_tokens=True)

                        file.write(generated_caption + "\n")

        print("Les légendes ont été générées avec succès et enregistrées dans le fichier de sortie.")

    except Exception as e:
        print("Une erreur s'est produite lors de la génération des légendes :", e)

# Fonction pour extraire les mots-clés à partir d'un fichier texte
def extract_keywords_from_text(file_path, output_file_path):
    with open(file_path, "r") as file:
        text = file.read()
    words = word_tokenize(text)
    # Filtrer les mots vides (stopwords)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Supprimer les doublons pour obtenir les mots-clés uniques
    keywords = list(set(filtered_words))

    # Créer une seule phrase à partir des mots-clés, séparés par des virgules
    summary_sentence = ', '.join(keywords)

    # Écrire les mots-clés dans le fichier de sortie
    with open(output_file_path, "w") as output_file:
        output_file.write(summary_sentence)

    print("Les mots-clés ont été écrits dans le fichier de sortie.")

# Fonction pour charger le modèle et prédire la violence
def predict_violence():
    # Chargement du modèle
    clf = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Charger les mots à partir du fichier
    with open('fichier.txt', 'r') as file:
        test_texts = file.read().split(',')

    # Nettoyer les espaces blancs autour des mots
    test_texts = [text.strip() for text in test_texts]

    # Créer une liste de tous les mots uniques dans votre jeu de données
    df = pd.read_csv('spam.csv', header=None, names=['label', 'text'])
    unique_words = set(df['text'])

    # Initialiser une liste pour stocker les mots associés à la violence
    violence_words = []

    # Prédictions pour les mots inconnus
    for text in test_texts:
        if text in unique_words:
            X_test = vectorizer.transform([text])
            prediction = clf.predict(X_test)
            if prediction[0] == 'Violence':
                violence_words.append(text)

    # Vérifier s'il y a des mots associés à la violence
    if violence_words:
        print("Résultat final : violence")
        print("Mots associés à la violence :", ', '.join(violence_words))
    else:
        print("Résultat final : nonviolence")

# Chemin de la vidéo d'entrée
video_path = "C:\\Users\\aser\\Desktop\\image_to text\\test\\Bharat-Intern-2nd-project-SMS-Classification-main\\datatest_vedio\\violence.mp4"

# Chemin du dossier de sortie pour les frames
output_folder_frames = "C:\\Users\\aser\\Desktop\\image_to text\\test\\Bharat-Intern-2nd-project-SMS-Classification-main\\dataset_vedio"

# Chemin du dossier de sortie pour les images significatives
output_folder = "C:\\Users\\aser\\Desktop\\image_to text\\test\\Bharat-Intern-2nd-project-SMS-Classification-main\\data"

# Chemin vers le fichier de sortie pour stocker les légendes
output_file = "C:\\Users\\aser\\Desktop\\image_to text\\test\\Bharat-Intern-2nd-project-SMS-Classification-main\\amin.txt"

# Convertir la vidéo en frames
input_folder_frames = video_to_frames(video_path, output_folder_frames)

# Filtrer les images significatives et générer des légendes
filter_and_generate_captions(input_folder_frames, output_folder, output_file)

# Chemin vers votre fichier texte
file_path = "C:\\Users\\aser\\Desktop\\image_to text\\test\\Bharat-Intern-2nd-project-SMS-Classification-main\\amin.txt"

# Chemin vers le fichier de sortie pour les mots-clés
output_file_path = "fichier.txt"

# Extraire les mots-clés du fichier texte
extract_keywords_from_text(file_path, output_file_path)

# Prédire la violence
predict_violence()
