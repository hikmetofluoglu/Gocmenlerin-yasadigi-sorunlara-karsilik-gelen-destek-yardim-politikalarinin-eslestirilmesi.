import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv(r"C:\Users\Hikmet Ofluoglu\Desktop\gocmen_sikayetleri_50mb.csv")
data = data.dropna()
data.head()

from collections import Counter

tokenized_words = word_tokenize(' '.join(data['sikayet_metni']))
frequency = Counter(tokenized_words)
most_common = frequency.most_common(1000)
ranks = np.arange(1, len(most_common)+1)
frequencies = [freq for _, freq in most_common]

plt.figure(figsize=(10, 6))
plt.loglog(ranks, frequencies)
plt.xlabel("Kelime Sırası (log)")
plt.ylabel("Frekans (log)")
plt.title("Zipf Yasası - Ham Veri")
plt.grid(True)
plt.show()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def lemmatize_text(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

def stem_text(tokens):
    return [stemmer.stem(token) for token in tokens]

data['tokens'] = data['sikayet_metni'].apply(preprocess_text)
data['lemmatized'] = data['tokens'].apply(lemmatize_text)
data['stemmed'] = data['tokens'].apply(stem_text)


output_dir = r'C:\Users\Hikmet Ofluoglu\Desktop\output'
os.makedirs(output_dir, exist_ok=True) 

data[['lemmatized']].to_csv(os.path.join(output_dir, 'lemmatized_data.csv'), index=False)
data[['stemmed']].to_csv(os.path.join(output_dir, 'stemmed_data.csv'), index=False)


for col in ['lemmatized', 'stemmed']:
    all_words = [word for tokens in data[col] for word in tokens]
    frequency = Counter(all_words)
    most_common = frequency.most_common(1000)
    ranks = np.arange(1, len(most_common)+1)
    frequencies = [freq for _, freq in most_common]

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies)
    plt.xlabel("Kelime Sırası (log)")
    plt.ylabel("Frekans (log)")
    plt.title(f"Zipf Yasası - {col.capitalize()} Veri")
    plt.grid(True)
    plt.show()


tfidf_lem = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)
tfidf_stem = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, token_pattern=None)

X_lem = tfidf_lem.fit_transform(data['lemmatized'])
X_stem = tfidf_stem.fit_transform(data['stemmed'])

pd.DataFrame(X_lem.toarray(), columns=tfidf_lem.get_feature_names_out()).to_csv(os.path.join(output_dir, "tfidf_lemmatized.csv"), index=False)
pd.DataFrame(X_stem.toarray(), columns=tfidf_stem.get_feature_names_out()).to_csv(os.path.join(output_dir, "tfidf_stemmed.csv"), index=False)


def get_most_similar_words(word, vectorizer, corpus):
    word_index = list(vectorizer.get_feature_names_out()).index(word) 
    vector = vectorizer.transform(corpus).toarray()[:, word_index]  
    
    similarities = cosine_similarity(vector.reshape(1, -1), vector) 
    most_similar_indices = similarities.argsort()[0][-6:-1][::-1]  
    most_similar_words = [corpus[i] for i in most_similar_indices]
    
    return most_similar_words

example_word = 'registration'
print("En yakın kelimeler:", get_most_similar_words(example_word, tfidf_lem, data['lemmatized']))
print("En yakın kelimeler (stemmed):", get_most_similar_words(example_word, tfidf_stem, data['stemmed']))
