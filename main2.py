import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import matplotlib.pyplot as plt

# settings (change these as needed)
PLOT_FILE = 'raw/plot_summaries.txt'
META_FILE = 'raw/movie.metadata.tsv'
CACHE_DIR = 'cache'
MIN_COUNT = 1500  # minimum genre frequency
USE_EMBEDDINGS = True

# setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# try to import embeddings
try:
    from sentence_transformers import SentenceTransformer
except:
    USE_EMBEDDINGS = False

# helper to read metadata
def load_meta():
    path = os.path.join(CACHE_DIR, 'meta.pkl')
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    meta = {}
    for line in open(META_FILE, encoding='utf-8'):
        parts = line.strip().split('\t')
        if len(parts) < 3:
            continue
        mid = parts[0]
        title = parts[2]
        raw = parts[-1]
        try:
            genres = list(json.loads(raw).values())
        except:
            genres = []
        meta[mid] = {'title': title, 'genres': genres}
    os.makedirs(CACHE_DIR, exist_ok=True)
    pickle.dump(meta, open(path, 'wb'))
    return meta

# helper to read summaries
def load_summaries(meta):
    path = os.path.join(CACHE_DIR, 'sum.pkl')
    if os.path.exists(path):
        try:
            return pickle.load(open(path, 'rb'))
        except:
            print('Corrupted cache, rebuilding')
    data = []
    for line in tqdm(open(PLOT_FILE, encoding='utf-8')):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        mid, text = parts[0], parts[1]
        info = meta.get(mid, {'title':'', 'genres':[]})
        # simple preprocess
        text = text.lower()
        text = ''.join(c for c in text if c.isalpha() or c.isspace())
        tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        data.append({
            'id': mid,
            'title': info['title'],
            'genres': info['genres'],
            'text': ' '.join(tokens)
        })
    pickle.dump(data, open(path, 'wb'))
    return data

# vectorize
def get_features(data):
    docs = [d['text'] for d in data]
    vec = TfidfVectorizer(max_df=0.8, min_df=5)
    X = vec.fit_transform(docs)
    return vec, X

# embeddings
def get_emb(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([d['text'] for d in data], show_progress_bar=True)

# labels
def get_labels(data):
    all_gen = [g for d in data for g in d['genres']]
    freq = pd.Series(all_gen).value_counts()
    keep = set(freq[freq>=MIN_COUNT].index)
    labels = [[g for g in d['genres'] if g in keep] for d in data]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    print('Kept genres:', len(mlb.classes_))
    return mlb, y

# train/eval
def train_eval(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(LinearSVC(class_weight='balanced'))
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    stats = {
        'micro F1': f1_score(yte, yp, average='micro'),
        'macro F1': f1_score(yte, yp, average='macro', zero_division=0),
        'precision': precision_score(yte, yp, average='micro'),
        'recall': recall_score(yte, yp, average='micro')
    }
    print('Stats:', stats)
    print(classification_report(yte, yp, target_names=mlb.classes_))
    return clf, stats

# recommend
def recommend(X, data, idx=0):
    nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(X)
    d, i = nbrs.kneighbors(X[idx].reshape(1, -1))
    recs = []
    for dist, j in zip(d[0][1:], i[0][1:]):
        recs.append({
            'title': data[j]['title'],
            'genres': data[j]['genres'],
            'score': float(1-dist)
        })
    return recs

# main
meta = load_meta()
movies = load_summaries(meta)

vec, X = get_features(movies)
pickle.dump(vec, open(os.path.join(CACHE_DIR, 'vec.pkl'), 'wb'))

if USE_EMBEDDINGS:
    X = get_emb(movies)
    pickle.dump(X, open(os.path.join(CACHE_DIR, 'emb.pkl'), 'wb'))

mlb, y = get_labels(movies)
model, stats = train_eval(X, y)
pickle.dump(model, open(os.path.join(CACHE_DIR, 'model.pkl'), 'wb'))

# plot metrics
plt.bar(stats.keys(), stats.values())
plt.title('Model Performance Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.show()

# top genres
plt.figure()
top10 = pd.Series([g for d in movies for g in d['genres']]).value_counts().head(10)
top10.plot(kind='bar')
plt.title('Top 10 Genres in Dataset')
plt.xlabel('Genre')
plt.ylabel('Frequency')
plt.show()

# sample rec and save outputs for random movies
import random
# pick 3 random movie indices
random.seed(42)
indices = random.sample(range(len(movies)), 3)
rec_list = []
for idx in indices:
    recs = recommend(X, movies, idx=idx)
    rec_list.append({
        'movie': movies[idx]['title'],
        'recommendations': recs
    })

# save metrics and recs
os.makedirs(CACHE_DIR, exist_ok=True)
with open(os.path.join(CACHE_DIR, 'metrics.json'), 'w') as f:
    json.dump(stats, f, indent=2)
with open(os.path.join(CACHE_DIR, 'recommendations.json'), 'w') as f:
    json.dump(rec_list, f, indent=2)

# print refined test outputs to console
print('=== Performance Metrics ===')
for k, v in stats.items():
    print(f"{k:12s}: {v:.3f}")

print('=== Sample Recommendations ===')
for item in rec_list:
    print(f"Movie: {item['movie']}")
    for rec in item['recommendations']:
        genres = rec.get('genres', [])
        genre_str = ', '.join(genres) if genres else 'None'
        print(f"  - {rec['title']} | Genres: {genre_str} | Score: {rec['score']:.2f}")
    print()

print(f"Metrics saved to {os.path.join(CACHE_DIR, 'metrics.json')}")
print(f"Recommendations saved to {os.path.join(CACHE_DIR, 'recommendations.json')}")

