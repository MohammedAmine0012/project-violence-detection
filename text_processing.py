import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def extract_keywords_from_text(text, output_file_path):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    keywords = list(set(filtered_words))
    summary_sentence = ', '.join(keywords)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(summary_sentence)

def extract_keywords_from_text_video(file_path, output_file_path):
    with open(file_path, "r") as file:
        text = file.read()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]

    keywords = list(set(filtered_words))
    summary_sentence = ', '.join(keywords)

    with open(output_file_path, "w") as output_file:
        output_file.write(summary_sentence)