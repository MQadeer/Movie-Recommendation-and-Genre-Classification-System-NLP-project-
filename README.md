# Movie-Recommendation-and-Genre-Classification-System-NLP-project-

# Methodology Deviations: Proposal vs Implementation Analysis

## Key Deviations from Original Proposal

### 1. Machine Learning Algorithm Change

| Aspect | Proposal | Implementation |
|--------|----------|----------------|
| Algorithm | Logistic Regression | Linear Support Vector Machine (LinearSVC) |
| Strategy | One-vs-Rest Logistic Regression | One-vs-Rest LinearSVC |

**Reasoning for Change:**

- **Better Performance**: SVM typically outperforms logistic regression on high-dimensional, sparse text data
- **Robustness**: LinearSVC is specifically optimized for text classification tasks
- **Class Imbalance**: The class_weight='balanced' parameter in SVM handles imbalanced genre distribution better than standard logistic regression
- **Scalability**: LinearSVC scales better with large vocabulary sizes common in NLP tasks
- **Research Evidence**: Literature shows SVM often achieves superior results for multi-label text classification

### 2. Feature Extraction Enhancement

| Aspect | Proposal | Implementation |
|--------|----------|----------------|
| Method | TF-IDF only | TF-IDF + Sentence Embeddings (dual approach) |
| Approach | Traditional bag-of-words | Modern transformer-based embeddings |

**Reasoning for Enhancement:**

- **Semantic Understanding**: Sentence embeddings capture semantic meaning beyond keyword matching
- **State-of-the-Art**: Transformer models (all-MiniLM-L6-v2) represent current best practices in NLP
- **Better Representations**: Dense embeddings often outperform sparse TF-IDF for similarity tasks
- **Flexibility**: Code allows switching between traditional and modern approaches
- **Future-Proofing**: The Embedding approach is more aligned with current NLP trends

### 3. Recommendation System Implementation

| Aspect | Proposal | Implementation |
|--------|----------|----------------|
| Method | Cosine Similarity on TF-IDF | Nearest Neighbors with Cosine Metric |
| Computation | Manual similarity computation | Scikit-learn's optimized NearestNeighbors |

**Reasoning for Change:**

- **Efficiency**: NearestNeighbors is highly optimized for similarity search
- **Scalability**: Better performance on large datasets
- **Flexibility**: Can easily change distance metrics or number of neighbors
- **Memory Efficiency**: Avoids computing the full pairwise similarity matrix
- **Industry Standard**: More commonly used in production recommendation systems




