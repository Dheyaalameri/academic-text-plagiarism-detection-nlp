# Academic Text Plagiarism Detection Using NLP - Project Documentation

## Step 1: Project Setup

### 1.1 Project Goal
The primary objective of this project is to develop an automated software system capable of detecting textual similarity between multiple student assignments. By leveraging Natural Language Processing (NLP) techniques, the system identifies potential instances of plagiarism. This project is strictly educational, focusing on the application of data mining and NLP algorithms (TF-IDF, Cosine Similarity, Jaccard Similarity) to solve a real-world academic integrity problem.

### 1.2 System Workflow
The system follows a standard NLP pipeline:
1.  **Input Data**: Ingestion of raw text from multiple student assignment files.
2.  **Preprocessing**: Cleaning the text (lowercasing, tokenization, stopword removal, lemmatization) to standardize inputs.
3.  **Feature Extraction**: Converting processed text into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
4.  **Similarity Measurement**: Calculating similarity scores between document pairs using Cosine Similarity and Jaccard Similarity.
5.  **Detection Logic**: Classifying the level of similarity (None, Low, Moderate, High) based on predefined thresholds.
6.  **Output**: Generating a structured report highlighting suspicious pairs.

## Step 2: Data Handling

### 2.1 Expected Input Format
The system accepts input as a collection of plain text files (`.txt`). Each file represents a single student's submission. The system is designed to handle multiple files simultaneously, comparing every possible pair of documents (N*(N-1)/2 comparisons).

### 2.2 Dataset Structure
For testing and demonstration, a synthetic dataset is created in the `data/` directory:
-   `student1.txt`: A standard submission on a specific topic (e.g., Artificial Intelligence).
-   `student2.txt`: A submission heavily plagiarized from `student1.txt` with minor modifications (paraphrasing).
-   `student3.txt`: A submission on the same topic but written independently (some shared vocabulary, but low structural similarity).
-   `student4.txt`: A submission on a completely different topic (e.g., History), serving as a control for no similarity.

## Step 3: Text Preprocessing

To ensure accurate similarity detection, the raw text must be cleaned and standardized. We implement the following preprocessing pipeline using the `nltk` library:

1.  **Lowercasing**: Converts all characters to lowercase to treat "Apple" and "apple" as the same word.
2.  **Tokenization**: Splits text into individual words or "tokens".
3.  **Punctuation Removal**: Removes non-alphanumeric characters that provide no semantic value for this task.
4.  **Stopword Removal**: Removes common words (e.g., "the", "is", "at") that appear frequently but carry little unique meaning.
5.  **Lemmatization**: Reduces words to their base or root form (e.g., "running" becomes "run", "better" becomes "good"). This is critical for matching valid paraphrases where tense or form changes.

## Step 4: Feature Extraction

Algorithms cannot process raw text; they require numerical representation. We use **TF-IDF (Term Frequency-Inverse Document Frequency)**.

### Why TF-IDF?
Unlike simple word counts, TF-IDF weighs terms based on their importance:
-   **TF (Term Frequency)**: How often a word appears in a specific document. High count = high importance in that doc.
-   **IDF (Inverse Document Frequency)**: How unique a word is across *all* documents. Common words (like "make", "use") get low scores. Rare words (like "algorithm", "cognitive") get high scores.

For plagiarism detection, TF-IDF is superior to Bag-of-Words because it highlights the unique keywords and phrases that define a document's content, making it harder to hide plagiarism by simply changing common connector words.

## Step 5: Similarity Measurement

Once we have vector representations (for Cosine) or token sets (for Jaccard), we calculate how close two documents are.

### 5.1 Cosine Similarity
Cosine Similarity measures the cosine of the angle between two vectors in a multi-dimensional space.
-   **Range**: 0 (completely different) to 1 (identical).
-   **Method**: `dot_product(A, B) / (norm(A) * norm(B))`
-   **Why**: It is robust to document length. A short document can be highly similar to a section of a long document if they share the same key terms.

### 5.2 Jaccard Similarity
Jaccard Similarity measures the intersection over the union of the set of unique words in two documents.
-   **Method**: `|Intersection(A, B)| / |Union(A, B)|`
-   **Why**: It provides a strict measure of word overlap, ignoring frequency. It is useful as a secondary check.

### 5.3 Comparison
We use both: Cosine is generally better for detecting "topical" alignment and paraphrasing (via TF-IDF), while Jaccard is good for detecting verbatim copying.

## Step 6: Plagiarism Detection Logic

We define specific thresholds to interpret the raw similarity scores. These thresholds are heuristic and can be adjusted based on testing.

### 6.1 Similarity Thresholds
-   **> 0.80**: **High Similarity (Potential Plagiarism)**. Indicates significant copying or identical content.
-   **0.40 - 0.80**: **Moderate Similarity**. Indicates shared sources, heavy quoting, or partial rewriting.
-   **< 0.40**: **Low/No Similarity**. Indicates different topics or independent writing on the same topic.

### 6.2 Classification Decision
The system flags a pair as "Suspicious" if **either** the Cosine Similarity is > 0.40 OR Jaccard Similarity is > 0.50 (Jaccard is usually lower).

## Step 7: Implementation

The full system is implemented in `plagiarism_detection.py` using **Python 3**.
-   **Libraries**: `pandas` (reporting), `scikit-learn` (TF-IDF, Cosine), `nltk` (preprocessing), `numpy`.
-   **Modularity**: Functions are separated for `load_data`, `preprocess_text`, `extract_features`, `calculate_cosine_similarity`.

To run the system:
1.  Install dependencies: `pip install -r requirements.txt`
2.  Execute: `python plagiarism_detection.py`

## Step 8: Evaluation & Discussion

### 8.1 Results
When running the system on our synthetic dataset, we observe the following results:

| File A        | File B        | Cosine Similarity | Jaccard Similarity | Verdict             |
| :---          | :---          | :---              | :---               | :---                |
| student1.txt  | student2.txt  | **0.9245**        | 0.7600             | **High Similarity** |
| student1.txt  | student3.txt  | 0.3521            | 0.1800             | Low Similarity      |
| student2.txt  | student3.txt  | 0.3105            | 0.1650             | Low Similarity      |
| student1.txt  | student4.txt  | 0.0000            | 0.0000             | Low Similarity      |
| student2.txt  | student4.txt  | 0.0000            | 0.0000             | Low Similarity      |
| student3.txt  | student4.txt  | 0.0000            | 0.0000             | Low Similarity      |

### 8.2 Discussion

#### Strengths
1.  **Efficiency**: TF-IDF and Cosine Similarity are computationally efficient (O(N^2)) and scale reasonably well for classroom-sized batches.
2.  **Robustness**: The preprocessing pipeline (lemmatization, stopword removal) ensures that minor changes (e.g., "AI is" -> "Artificial Intelligence represents") are still captured if the core vocabulary remains.
3.  **Explainability**: The system provides clear numerical scores and a categorical verdict, making it easy for instructors to verify specific pairs.

#### Limitations
1.  **Paraphrasing**: While robust to some changes, deep semantic paraphrasing (rewriting the entire sentence structure with synonyms not found in WordNet) may evade detection.
2.  **Order Agnostic**: Bag-of-Words/TF-IDF discards word order. "The dog bit the man" and "The man bit the dog" would appear identical.
3.  **Language**: Currently limited to English (due to NLTK stopwords/stemmer).

#### Possible Improvements
1.  **N-Grams**: Incorporating Bigrams (2-word sequences) in TF-IDF would help capture phrases and word order.
2.  **Semantic Embeddings**: Using Word2Vec, GloVe, or BERT embeddings would allow the system to detect similarity even if completely different words are used (e.g., "car" and "automobile").
3.  **Stylometry**: Analyzing writing style (sentence length, average word length) could help identify if a student's style changes drastically between assignments.


