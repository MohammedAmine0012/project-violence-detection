import joblib
import pandas as pd

def predict_violence(keywords_file):
    clf = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    with open(keywords_file, 'r', encoding='utf-8') as file:
        test_texts = file.read().split(',')

    test_texts = [text.strip() for text in test_texts]

    df = pd.read_csv('spam.csv', header=None, names=['label', 'text'])
    unique_words = set(df['text'])

    violence_types = []
    violence_words = []

    for text in test_texts:
        if text in unique_words:
            X_test = vectorizer.transform([text])
            prediction = clf.predict(X_test)
            if prediction[0] == 'Violence':
                violence_words.append(text)
                # Ajoutez ici la logique pour déterminer le type de violence
                violence_type = determine_violence_type(text)
                violence_types.append(violence_type)

    if violence_words:
        result = "Violence detected"
        return result, violence_types
    else:
        result = "No violence detected"
        return result, []

def determine_violence_type(text):
    # Convertir le texte en minuscules pour une correspondance insensible à la casse
    text_lower = text.lower()

    # Liste de mots-clés associés à différents types de violence
    keywords_physical_violence = ['frapper','fighting','touching']
    keywords_firearm_violence = ['arme à feu','tirer', 'pistolet', 'fusil']
    keywords_knife_violence = ['knife', 'pointing', 'holding','couteau','poignarder', 'lame', 'égorger']

    # Vérifier la présence de mots-clés pour chaque type de violence
    if any(keyword in text_lower for keyword in keywords_physical_violence):
        return "Physical violence"
    elif any(keyword in text_lower for keyword in keywords_firearm_violence):
        return "Firearm"
    elif any(keyword in text_lower for keyword in keywords_knife_violence):
        return "White weapon"
    else:
        return "Type de violence non specifie"

def predict_violence_video():
    clf = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    with open('fichier.txt', 'r') as file:
        test_texts = file.read().split(',')

    test_texts = [text.strip() for text in test_texts]

    df = pd.read_csv('spam.csv', header=None, names=['label', 'text'])
    unique_words = set(df['text'])

    violence_words = []
    violence_types = set()  # Utilisation d'un ensemble pour stocker les types de violence uniques

    for text in test_texts:
        if text in unique_words:
            X_test = vectorizer.transform([text])
            prediction = clf.predict(X_test)
            if prediction[0] == 'Violence':
                violence_words.append(text)
                violence_type = determine_violence_type(text)
                violence_types.add(violence_type)  # Ajout du type de violence à l'ensemble

    if violence_words:
        print("Violence detected")
        print("Types of violence detected :", ', '.join(violence_types))
    else:
        print("Non-violence")
