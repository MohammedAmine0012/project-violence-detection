import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Charger les données à partir du fichier CSV
df = pd.read_csv('spam.csv', header=None, names=['label', 'text'])

# Séparer les données en features (texte) et labels (violence/nonviolence)
X = df['text']
y = df['label']

# Vectorisation du texte
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Entraînement du modèle SVM
clf = SVC(kernel='linear')
clf.fit(X, y)

# Sauvegarde du modèle
joblib.dump(clf, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
