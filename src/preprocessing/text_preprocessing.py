import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('french'))  # Vous pouvez choisir une autre langue si nécessaire

    def preprocess_text(self, text):
        # Supprimer les balises HTML
        text = BeautifulSoup(text, 'html.parser').get_text()

        # Supprimer les caractères non alphabétiques
        text = re.sub(r'[^a-zA-Z]', ' ', text)

        # Tokenization
        words = word_tokenize(text.lower())

        # Suppression des stopwords et lemmatisation
        filtered_words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        return ' '.join(filtered_words[:10])

    def preprocess_text_in_df(self, df, columns):
        for column in columns:
            df[column] = df[column].apply(self.preprocess_text)

