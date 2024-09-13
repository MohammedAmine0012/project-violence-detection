import os

# Définition de la fonction pour vider le contenu d'un fichier
def clear_file_content(file_path):
    open(file_path, 'w').close()

# Définition de la fonction pour vider le contenu d'un dossier
def clear_folder_content(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Erreur lors de la suppression du fichier {file_path} : {e}")
