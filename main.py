from violence_detection import filter_random
from video_processing import video_to_frames
from filter_and_generate_captions import filter_and_generate_captions
from text_processing import extract_keywords_from_text_video
from model_prediction import predict_violence_video
from file_operations import clear_file_content, clear_folder_content
import sys

def main(video_path):
    # Chemin du dossier de sortie pour les frames
    output_folder_frames = "dataset_video"

    # Chemin du dossier de sortie pour les images significatives
    output_folder = "data"

    # Chemin vers le fichier de sortie pour stocker les légendes
    output_file = "amin.txt"

    # Convertir la vidéo en frames
    input_folder_frames = video_to_frames(video_path, output_folder_frames)

    # Filtrer les images significatives et générer des légendes
    filter_and_generate_captions(input_folder_frames, output_folder, output_file)

    # Chemin vers votre fichier texte
    file_path = output_file

    # Chemin vers le fichier de sortie pour les mots-clés
    output_file_path = "fichier.txt"

    # Extraire les mots-clés du fichier texte
    extract_keywords_from_text_video(file_path, output_file_path)
    
    # Prédire la violence
    result = predict_violence_video()
    return result

if __name__ == "__main__":
    # Vérifier s'il y a au moins un argument (le premier argument est le nom du script)
    if len(sys.argv) < 2:
        print("Veuillez fournir le chemin de la vidéo en argument de ligne de commande.")
        sys.exit(1)
    
    # Récupérer le chemin de la vidéo à partir des arguments de la ligne de commande
    video_path = sys.argv[1]

    # Exécuter le script principal avec le chemin de la vidéo
    result = main(video_path)
