import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, ndcg_score
import pickle, os
import requests
from imdb import IMDb
import numpy as np, random

# IMDb client for fetching ground-truth recs
ia = IMDb()

# Load movie metadata for title lookup
with open('movie_metadata.pkl', 'rb') as f:
    movie_metadata = pickle.load(f)

# Build mapping from numeric summary ID to Freebase MID using TSV
id_to_fbmid = {}
with open(os.path.join('raw', 'movie.metadata.tsv'), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            num_id, fb_mid = parts[0], parts[1]
            id_to_fbmid[num_id] = fb_mid

# Fetch IMDb's 'More Like This' recommendations for a given IMDb ID
def get_imdb_recs(imdb_id, top_n=10):
    """
    Fetch IMDb 'More Like This' recs, stripping 'tt' prefix for IMDbPY compatibility.
    """
    if not imdb_id:
        return []
    # extract numeric part
    digits = ''.join(filter(str.isdigit, imdb_id))
    if not digits:
        return []
    try:
        movie = ia.get_movie(digits)
    except Exception:
        return []
    recs = movie.get('recommendations', [])[:top_n] or []
    return [r.movieID for r in recs]

# Ranking metrics
def precision_at_k(pred_ids, gt_ids, k):
    return len(set(pred_ids[:k]) & set(gt_ids)) / k if k else 0

def recall_at_k(pred_ids, gt_ids, k):
    return len(set(pred_ids[:k]) & set(gt_ids)) / len(gt_ids) if gt_ids else 0

# Map Freebase MID → IMDb ID via Wikidata SPARQL (P646 → P345)
def get_imdb_id_from_freebase(fb_mid):
    query = f'''
    SELECT ?imdb WHERE {{
      ?movie wdt:P646 "{fb_mid}".
      ?movie wdt:P345 ?imdb.
    }}
    '''
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "nlp-project/1.0 (contact@example.com)"
    }
    r = requests.get(url, params={"query": query}, headers=headers)
    data = r.json()
    bindings = data.get("results", {}).get("bindings", [])
    if not bindings:
        return None
    return bindings[0]["imdb"]["value"]

# Build ordered list of summary IDs from raw plot_summaries.txt
summary_ids = []
with open(os.path.join('raw', 'plot_summaries.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if parts:
            summary_ids.append(parts[0])
# Build list of Freebase MIDs in the same order as summary_ids
summary_fb_mids = [id_to_fbmid.get(sid) for sid in summary_ids]

def get_top_similar_movies(summary_index, tf_idf_sparse_matrix_path, top_n=10):
    """
    Calculates the cosine similarity between a movie's TF-IDF vector and all other movies,
    and returns the indices of the top N most similar movies.

    Args:
        summary_index (int): The index of the movie summary to compare against.
        tf_idf_sparse_matrix_path (str): Path to the TF-IDF sparse matrix in .npz format.
        top_n (int): The number of top similar movies to return.

    Returns:
        List[str]: Titles of the top N most similar movies, excluding the input movie itself.
    """

    # Load the TF-IDF sparse matrix
    tf_idf_matrix = load_npz(tf_idf_sparse_matrix_path)

    # Get the vector for the given summary index
    summary_vector = tf_idf_matrix[summary_index]

    # Calculate cosine similarity between the summary vector and all other vectors
    similarity_scores = cosine_similarity(summary_vector, tf_idf_matrix).flatten()

    # Get the indices of the top N most similar movies
    # Exclude the input movie itself by setting its similarity to -1
    similarity_scores[summary_index] = -1  # Ensure the movie itself is not included
    top_movie_indices = np.argsort(similarity_scores)[::-1][0:top_n]

    return top_movie_indices

# Helper to fetch movie title from IMDb given an IMDb ID string (digits)
def get_imdb_title(imdb_id):
    digits = ''.join(filter(str.isdigit, str(imdb_id)))
    if not digits:
        return ''
    try:
        movie = ia.get_movie(digits)
        return movie.get('title', '')
    except Exception:
        return ''

if __name__ == '__main__':
    # Build test split of movie indices (e.g. 100 random movies)
    tfidf_path = 'tf_idf_sparse_matrix.npz'
    num_movies = len(summary_ids)
    random.seed(42)
    test_idxs = random.sample(range(num_movies), min(100, num_movies))
    top_n = 10

    # Map internal indices to IMDb IDs
    internal_to_imdb = {}
    for idx in test_idxs:
        fb_mid = summary_fb_mids[idx]
        internal_to_imdb[idx] = get_imdb_id_from_freebase(fb_mid)

    # Build global genre list
    genre_set = set()
    for sid in summary_ids:
        genre_set.update(movie_metadata.get(sid, {}).get('genres', []))
    genre_list = sorted(genre_set)

    # Containers for extrinsic ranking evaluation
    all_precisions = {k: [] for k in (1,5,10)}
    all_recalls    = {k: [] for k in (1,5,10)}
    mrrs, ndcgs    = [], []
    # Containers for intrinsic classification evaluation
    y_true, y_pred = [], []

    for idx in test_idxs:
        # 1) Recommendation evaluation
        imdb_id = internal_to_imdb[idx]
        gt = set(get_imdb_recs(imdb_id, top_n=top_n))
        preds = get_top_similar_movies(idx, tfidf_path, top_n=top_n)
        for k in (1,5,10):
            p_at_k = len(set(preds[:k]) & gt) / k
            r_at_k = len(set(preds[:k]) & gt) / len(gt) if gt else 0
            all_precisions[k].append(p_at_k)
            all_recalls[k].append(r_at_k)
        # MRR
        def reciprocal_rank(gt, preds):
            for rank, p in enumerate(preds, 1):
                if p in gt:
                    return 1/rank
            return 0
        mrrs.append(reciprocal_rank(gt, preds))
        # NDCG@10
        rel = [1 if p in gt else 0 for p in preds]
        ndcgs.append(ndcg_score([rel], [rel], k=10))
        # 2) Genre-classification evaluation (intrinsic)
        true_genres = set(movie_metadata.get(summary_ids[idx], {}).get('genres', []))
        # TODO: replace this stub with actual genre-prediction logic
        pred_genres = true_genres
        y_true.append([1 if g in true_genres else 0 for g in genre_list])
        y_pred.append([1 if g in pred_genres else 0 for g in genre_list])

    # Print ranking metrics
    for k in (1,5,10):
        print(f"P@{k}: {np.mean(all_precisions[k]):.3f}, R@{k}: {np.mean(all_recalls[k]):.3f}")
    print(f"Mean MRR: {np.mean(mrrs):.3f}")
    print(f"Mean NDCG@10: {np.mean(ndcgs):.3f}")

    # Compute multi-label genre-classification metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    micro_f1     = f1_score(y_true, y_pred, average='micro', zero_division=0)
    per_gen_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_gen_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)
    mean_jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)

    print("\nGenre Classification:")
    print(f" Micro-F1: {micro_f1:.3f}, Jaccard: {mean_jaccard:.3f}")
    for i, g in enumerate(genre_list):
        print(f" {g:15s} P: {per_gen_prec[i]:.2f}  R: {per_gen_rec[i]:.2f}")

