from typing import List
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
import json
import pickle
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
from typing import Set
from tqdm import tqdm
from nltk.corpus import stopwords

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.neighbors import NearestNeighbors




# ==============================================================================
# --- Text Preprocessing Functions ---
# ==============================================================================

# Download stopwords if not available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Downloads for Lemmatization
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Initialize global lemmatizer
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map NLTK POS tags to WordNet POS tags for accurate lemmatization."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def tokenzie(text: str) -> List[str]:
    """Tokenizes and cleans text, including number/punctuation removal, lowercasing, stopword filtering, and lemmatization."""
    text = re.sub(r"\d+","",text) # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation)) # Remove punctuation
    text = text.lower() # Convert to lowercase
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words] # Filter out non-alphabetic and stop words
    lemmatized_tokens = [lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens] # Lemmatize tokens
    return lemmatized_tokens

def createVocabulary(words: List[str]) -> List[str]:
    """Creates a sorted list of unique words from a list of tokens."""
    vocabulary = sorted(set(words))
    return list(vocabulary)



# ==============================================================================
# --- Movie Metadata Loading / Creation ---
# ==============================================================================

print("\n--- Starting Movie Metadata Processing ---")

movie_metadata_path = "movie_metadata.pkl"
movie_metadata = {}

if os.path.exists(movie_metadata_path):
    with open(movie_metadata_path, "rb") as f:
        movie_metadata = pickle.load(f)
    print("Movie metadata loaded from .pkl file.")
else:
    print("Movie metadata .pkl not found. Creating from .tsv...")
    with open ("raw/movie.metadata.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            movie_id = parts[0]
            title = parts[2] if len(parts) > 2 else ""
            genres_raw = parts[-1]
            try:
                genres = list(json.loads(genres_raw).values())
            except:
                genres = []
            movie_metadata[movie_id] = {
                "title": title,
                "genres": genres
            }
    with open(movie_metadata_path, "wb") as f:
        pickle.dump(movie_metadata, f)
    print("movie_metadata created")

print("--- Movie Metadata Processing Complete ---")



# ==============================================================================
# --- Summary List Creation ---
# ==============================================================================

print("\n--- Starting Summary List Processing ---")

@dataclass
class Summary:
    """Dataclass to hold all relevant information for a movie summary."""
    id: str
    title: str
    genres: List[str]
    text: str
    tokens: List[str]
    term_freqs: Counter
    vocabulary: Set[str]

summaries_path = "summaries.pkl"
summaries: List[Summary] = []

if os.path.exists(summaries_path):
    with open("summaries.pkl", "rb") as f:
        summaries = pickle.load(f)
    print("Summaries loaded from .pkl file.")
else:
    with open ("raw/plot_summaries.txt", "r", encoding="utf-8") as f:
        print("Summaries .pkl not found. Creating from plot summaries and metadata...")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            id_ = parts[0]
            text = parts[1] if len(parts) > 1 else ""

            title = movie_metadata.get(id_, {}).get("title", "")
            genres = movie_metadata.get(id_, {}).get("genres", [])
            tokens = tokenzie(text)
            term_freqs = Counter(tokens)
            s_vocabulary = set(tokens)

            s = Summary(
                id = id_,
                title = title,
                genres = genres,
                text = text,
                tokens = tokens,
                term_freqs = term_freqs,
                vocabulary = s_vocabulary
            )
            summaries.append(s)
    with open("summaries.pkl", "wb") as f:
        pickle.dump(summaries, f)
    print("summaries.pkl created.")

print(f"Total number of summaries processed: {len(summaries)}")
print("--- Summary List Processing Complete ---")



# ==============================================================================
# --- Filtering small summaries ---
# ==============================================================================

MIN_SUMMARY_TOKENS = 100

print(f"\n--- Filtering Short Summaries (Threshold: {MIN_SUMMARY_TOKENS} tokens) ---")
initial_summary_count = len(summaries)
print(f"Total summaries before length filtering: {initial_summary_count}")

filtered_summaries_by_length = []
skipped_by_length_count = 0

for s in summaries:
    if len(s.tokens) >= MIN_SUMMARY_TOKENS:
        filtered_summaries_by_length.append(s)
    else:
        skipped_by_length_count += 1

summaries = filtered_summaries_by_length

print(f"Summaries filtered out due to short length (< {MIN_SUMMARY_TOKENS} tokens): {skipped_by_length_count}")
print(f"Total summaries after filtering by length: {len(summaries)}")
print("--- Short Summary Filtering Complete ---")

# --- Saving the updated summaries (after length filtering) ---
with open("summaries_length_filtered.pkl", "wb") as f:
    pickle.dump(summaries, f)
print("Updated summaries_length_filtered.pkl saved (after length filtering).")



# ==============================================================================
# --- Filter Summaries by Genre and Prepare Final Data Set ---
# ==============================================================================

print("\n--- Filtering Summaries by Genre Exclusions ---")

EXCLUDED_GENRES = [
    # Reasons for exclusion:
    # - "Format/Style-based": Genres that describe the film's format or style rather than its primary thematic content.
    #   (e.g., "Black-and-white", "Short Film", "Silent film", "Computer Animation", "Stop motion", "Art film", "Experimental film")
    # - "Origin/Cultural-specific": Genres tied to a specific country or cultural background, potentially leading to bias or
    #   less generalizable recommendations across a broader movie dataset if not handled specifically.
    #   (e.g., "World cinema", "Japanese Movies", "Bollywood", "Chinese Movies", "Filipino Movies")
    # - "Production/Budget-based": Categories related to production scale or type, not core narrative.
    #   (e.g., "Indie", "Television movie", "B-movie")
    # - "Niche/Meta/Very Specific Content": Genres that are too niche, describe the film's relationship to other media,
    #   or refer to very specific sub-categories that might not be well-represented or distinct enough in summaries.
    #   (e.g., "Film adaptation", "Cult", "Ensemble Film", "Pre-Code", "Adult", "Christian film", "Social issues")
    # - "Overlapping/Broad Categories": Categories that are often covered by other, more specific genres
    #   or are too broad to provide distinct information for a content-based system.
    #   (e.g., "Documentary", "Animation", "Anime" - often have more specific sub-genres that are also present)
    # - "Non-thematic/Contextual": Genres describing the context of the film rather than its narrative themes.
    #   (e.g., "Culture & Society")
    "World cinema", "Black-and-white", "Indie", "Short Film", "Animation",
    "Japanese Movies", "Film adaptation", "Documentary", "Silent film",
    "Bollywood", "Chinese Movies", "Cult", "Television movie", "B-movie",
    "Art film", "Ensemble Film", "Anime", "Filipino Movies",
    "Culture & Society", "Computer Animation", "Stop motion", "Pre-Code",
    "Adult", "Christian film", "Experimental film", "Social issues"
]

final_summaries: List[Summary] = []

for s in summaries:
    valid_genres_for_summary = [genre for genre in s.genres if genre not in EXCLUDED_GENRES]
    if valid_genres_for_summary: # Only keep summaries that still have valid genres
        s.genres = valid_genres_for_summary
        final_summaries.append(s)

summaries = final_summaries # Update 'summaries' to the final filtered list

print(f"Summaries filtered out due to genre exclusion: {initial_summary_count - len(summaries)}")
print(f"Total summaries after all filtering (length and genre): {len(summaries)}")
print("--- Genre Filtering Complete ---")


# --- Saving the final summaries (after all filtering) ---
with open("summaries_final.pkl", "wb") as f:
    pickle.dump(summaries, f)
print("Final summaries_final.pkl saved (after all filtering).")


# ==============================================================================
# --- Vocabulary Creation (AFTER ALL Summary Filtering) ---
# ==============================================================================

print("\n--- Starting Vocabulary Creation from Filtered Summaries ---")

vocabulary_path = "vocabulary_filtered.pkl"
vocabulary: List[str] = []

if os.path.exists(vocabulary_path):
    with open(vocabulary_path, "rb") as f:
        vocabulary = pickle.load(f)
    print("Filtered vocabulary loaded from .pkl file.")
else:
    print("Filtered vocabulary .pkl not found. Creating from filtered summaries...")
    all_filtered_tokens = []
    for s in summaries:
        all_filtered_tokens.extend(s.tokens)

    vocabulary = createVocabulary(all_filtered_tokens)

    with open(vocabulary_path, "wb") as f:
        pickle.dump(vocabulary, f)
    print("Filtered vocabulary created and saved to .pkl file.")

print(f"Vocabulary Length after filtering summaries: {len(vocabulary)} words")
print("First 50 words of the Filtered Vocabulary:")
for i, token in enumerate(vocabulary[:50]):
    print(f"  {i}: {token}")

print("--- Vocabulary Creation from Filtered Summaries Complete ---")


# ==============================================================================
# --- Inverse Document Frequency (IDF) Calculation ---
# ==============================================================================

print("\n--- Starting IDF Calculation ---")

idf_path = "idf_dict.pkl"
idf_dict: dict[str, float] = {}

if os.path.exists(idf_path):
    with open (idf_path, "rb") as f:
        idf_dict = pickle.load(f)
    print("IDF dictionary loaded from .pkl file.")
else:
    print("IDF dictionary .pkl not found. Calculating IDF values...")

    n = len(summaries)

    # Step 1: Efficiently calculate document frequencies
    print(f"Calculating document frequencies for {len(summaries)} summaries...")
    doc_frequencies: Counter = Counter()
    for s in tqdm(summaries, desc="Counting DFs"):
        for token in s.vocabulary:
            doc_frequencies[token] += 1

    # Step 2: Calculate IDF values based on document frequencies
    print(f"Calculating IDF values for {len(vocabulary)} words...")
    for token in tqdm(vocabulary, desc="Calculating IDFs"):
        df = doc_frequencies.get(token, 0)
        idf = math.log(n / (df + 1)) # Add 1 to avoid division by zero for unseen words
        idf_dict[token] = idf

    with open(idf_path, "wb") as f:
        pickle.dump(idf_dict, f)
    print("IDF dictionary created and saved to .pkl file.")

    # Optionally save as JSON (rounded for readability/debugging)
    with open("idf_dict.json", "w", encoding="utf-8") as f:
        idf_dict_rounded = {k: round(v, 4) for k, v in idf_dict.items()}
        json.dump(idf_dict_rounded, f, indent=2)

print("--- IDF Calculation Complete ---")


# ==============================================================================
# --- TF-IDF Sparse Matrix Creation and Storage ---
# ==============================================================================

print("\n--- Starting TF-IDF Sparse Matrix Processing ---")

# Create a mapping from vocabulary words to their column indices in the matrix
vocabulary_to_idx = {word: i for i, word in enumerate(vocabulary)}
num_docs = len(summaries)
num_vocab = len(vocabulary)

tf_idf_sparse_matrix_path = "tf_idf_sparse_matrix.npz"
tf_idf_sparse_matrix = None

if os.path.exists(tf_idf_sparse_matrix_path):
    # Load sparse matrix in CSR format (more efficient for computations)
    tf_idf_sparse_matrix = load_npz(tf_idf_sparse_matrix_path)
    print("TF-IDF sparse matrix loaded from .npz file.")
else:
    print("TF-IDF sparse matrix .npz not found. Creating new matrix...")

    # Initialize a LIL (List of Lists) matrix for efficient incremental filling
    tf_idf_sparse_matrix = lil_matrix((num_docs, num_vocab), dtype=np.float32)

    # Iterate through each summary (row in the matrix)
    for doc_idx, s in tqdm(enumerate(summaries), total=num_docs, desc="Create TF-IDF Matrix"):
        # Iterate through words that actually appear in the current summary
        for word, tf_raw in s.term_freqs.items():
            if word in vocabulary_to_idx: # Ensure the word is in the (filtered) vocabulary
                word_idx = vocabulary_to_idx[word]
                idf = idf_dict.get(word, 0)

                # Apply log normalization for TF
                if tf_raw > 0:
                    tf_scaled = 1 + math.log(tf_raw)
                else:
                    tf_scaled = 0

                tf_idf = np.float32(tf_scaled * idf)
                if tf_idf > 0: # Only store non-zero TF-IDF values
                    tf_idf_sparse_matrix[doc_idx, word_idx] = tf_idf

    # Convert the LIL-Matrix to CSR (Compressed Sparse Row) format for efficient operations
    tf_idf_sparse_matrix = tf_idf_sparse_matrix.tocsr()
    save_npz(tf_idf_sparse_matrix_path, tf_idf_sparse_matrix)
    print(f"TF-IDF Sparse Matrix created")

# Display the dimensions of the TF-IDF matrix
print(f"Dimension of TF-IDF Sparse Matrix: {tf_idf_sparse_matrix.shape}")
print("--- TF-IDF Sparse Matrix Processing Complete ---")


# ==============================================================================
# --- Prepare Labels (y) for Multi-Label Classification ---
# ==============================================================================

print("\n--- Starting Multi-Threshold Model Evaluation ---")

# Extract genre labels directly from the already filtered summaries
genre_labels = [s.genres for s in summaries]

# Count the frequency for each genre
all_flat_genres_raw = [genre for sublist in genre_labels for genre in sublist]
genre_counts_raw = Counter(all_flat_genres_raw)

print(f"Original number of unique genres (before filtering): {len(genre_counts_raw)}")
print(f"Top 40 most common genres (before filtering) and their counts:")
for genre, count in genre_counts_raw.most_common(40):
    print(f"  - {genre}: {count}")
print("-" * 50)

# Define threshold values for the loop
threshold_values = []
threshold_values.extend(range(0, 501, 50))
threshold_values.extend(range(600, 1001, 100))
threshold_values.extend(range(1250, 2001, 250))

results = []

for MIN_GENRE_FREQUENCY in threshold_values:
    print(f"\n===== Evaluating with MIN_GENRE_FREQUENCY = {MIN_GENRE_FREQUENCY} =====")

    frequent_genres = {genre for genre, count in genre_counts_raw.items() if count >= MIN_GENRE_FREQUENCY}

    print(f"Number of genres after filtering (min frequency >= {MIN_GENRE_FREQUENCY}): {len(frequent_genres)}")

    current_iteration_genre_labels = []
    for movie_genres_list in genre_labels:
        current_movie_filtered_genres = [g for g in movie_genres_list if g in frequent_genres]
        current_iteration_genre_labels.append(current_movie_filtered_genres)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(current_iteration_genre_labels)

    print(f"Total unique genres after filtering: {len(mlb.classes_)}")

    if len(mlb.classes_) == 0:
        print(f"No genres remaining for MIN_GENRE_FREQUENCY = {MIN_GENRE_FREQUENCY}. Skipping training.")
        results.append({
            'threshold': MIN_GENRE_FREQUENCY, 'num_genres': 0, 'micro_f1': 0.0,
            'macro_f1': 0.0, 'micro_precision': 0.0, 'micro_recall': 0.0
        })
        continue

    # Verify consistency of samples before splitting
    if tf_idf_sparse_matrix.shape[0] != y.shape[0]:
        print(f"!!! Inconsistency Warning: TF-IDF matrix has {tf_idf_sparse_matrix.shape[0]} samples, but y has {y.shape[0]} samples.")
        print("This should not happen with the current code structure if filtering was done correctly.")
        raise ValueError("Sample count mismatch between TF-IDF matrix and labels (y).")


    X_train, X_test, y_train, y_test = train_test_split(
        tf_idf_sparse_matrix, y, test_size=0.2, random_state=42
    )
    print(f"Shape of X_train (training features): {X_train.shape}")
    print(f"Shape of X_test (testing features): {X_test.shape}")
    print(f"Shape of y_train (training labels): {y_train.shape}")
    print(f"Shape of y_test (testing labels): {y_test.shape}")

    base_classifier = LogisticRegression(solver='liblinear', max_iter=500, random_state=42)
    model = OneVsRestClassifier(base_classifier)

    print("Training the OneVsRest Logistic Regression model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    y_pred = model.predict(X_test)

    print("\n--- Model Evaluation ---")
    print(f"Used Genre Frequency Threshold: {MIN_GENRE_FREQUENCY}")
    print(f"Number of Genres Considered after Filtering: {len(mlb.classes_)}")

    micro_f1 = f1_score(y_test, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    micro_precision = precision_score(y_test, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_test, y_pred, average='micro', zero_division=0)

    print(f"Micro F1-Score: {micro_f1:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")

    results.append({
        'threshold': MIN_GENRE_FREQUENCY,
        'num_genres': len(mlb.classes_),
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall
    })

print("\n--- All Thresholds Evaluation Complete ---")

output_results_path = "model_evaluation_results.json"
with open(output_results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)
print(f"Model evaluation results saved to {output_results_path}")

print("\n--- Collected Results Across Thresholds ---")
for res in results:
    print(f"Threshold: {res['threshold']:<6}, Genres: {res['num_genres']:<4}, Micro F1: {res['micro_f1']:.4f}, Macro F1: {res['macro_f1']:.4f}, Micro P: {res['micro_precision']:.4f}, Micro R: {res['micro_recall']:.4f}")