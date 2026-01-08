import os
import string
import warnings
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# PROJECT CONFIGURATION (STEP 1 & 2)
# =============================================================================
DATA_DIR = 'data'
EXTENSIONS = ['.txt']

# Download necessary NLTK data
def download_nltk_resources():
    required_packages = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{package}')
            except LookupError:
                nltk.download(package, quiet=True)

download_nltk_resources()

# =============================================================================
# TEXT PREPROCESSING (STEP 3)
# =============================================================================
def preprocess_text(text: str) -> str:
    """
    Cleans and preprocesses the input text:
    1. Lowercasing
    2. Tokenization
    3. Punctuation removal
    4. Stopword removal
    5. Lemmatization
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Punctuation Removal & Alphanumeric check
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    
    # 4. Stopword Removal
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(lemmatized_words)

def load_data(directory: str) -> Tuple[List[str], List[str]]:
    """
    Loads text files from the specified directory.
    Returns a list of filenames and a list of file contents.
    """
    filenames = []
    documents = []
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created. Please add text files.")
        return [], []
        
    for filename in os.listdir(directory):
        if any(filename.endswith(ext) for ext in EXTENSIONS):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                filenames.append(filename)
                documents.append(content)
                
    return filenames, documents

# =============================================================================
# FEATURE EXTRACTION (STEP 4)
# =============================================================================
def extract_features(documents: List[str]):
    """
    Converts a list of preprocessed documents into a TF-IDF matrix.
    Returns the vectorizer and the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# =============================================================================
# SIMILARITY MEASUREMENT (STEP 5)
# =============================================================================
def calculate_cosine_similarity(tfidf_matrix):
    """
    Calculates the Cosine Similarity matrix for the documents.
    """
    return cosine_similarity(tfidf_matrix)

def calculate_jaccard_similarity(doc1: str, doc2: str) -> float:
    """
    Calculates Jaccard Similarity between two preprocessed text strings.
    J(A,B) = |intersection| / |union|
    """
    set1 = set(doc1.split())
    set2 = set(doc2.split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

# =============================================================================
# PLAGIARISM DETECTION LOGIC (STEP 6)
# =============================================================================
def classify_similarity(cosine_score: float, jaccard_score: float) -> str:
    """
    Classifies the similarity based on thresholds.
    Thresholds:
    - High: > 0.8
    - Moderate: > 0.4
    - Low: <= 0.4
    """
    # Using primarily Cosine for classification, Jaccard as supporting metric
    score = cosine_score
    
    if score > 0.8:
        return "High Similarity"
    elif score > 0.4:
        return "Moderate Similarity"
    else:
        return "Low Similarity"

# =============================================================================
# MAIN IMPLEMENTATION (STEP 7)
# =============================================================================
if __name__ == "__main__":
    print("-" * 60)
    print("      ACADEMIC TEXT PLAGIARISM DETECTION SYSTEM")
    print("-" * 60)
    
    # 1. Load Data
    filenames, raw_docs = load_data(DATA_DIR)
    
    if not raw_docs or len(raw_docs) < 2:
        print("Error: Need at least 2 text files in 'data/' directory to compare.")
    else:
        print(f"Loaded {len(raw_docs)} files: {filenames}\n")
        
        # 2. Preprocessing
        print("Step 1: Preprocessing Texts...")
        processed_docs = [preprocess_text(doc) for doc in raw_docs]
        
        # 3. Feature Extraction (TF-IDF)
        print("Step 2: Extracting Features (TF-IDF)...")
        vectorizer, tfidf_matrix = extract_features(processed_docs)
        
        # 4. Calculate Similarity
        print("Step 3: Calculating Similarity Scores...\n")
        cosine_sim_matrix = calculate_cosine_similarity(tfidf_matrix)

        # 5. Generate Report
        results = []
        
        # Iterate through unique pairs
        num_files = len(filenames)
        for i in range(num_files):
            for j in range(i + 1, num_files):
                file1 = filenames[i]
                file2 = filenames[j]
                
                # Cosine Score
                cosine_score = cosine_sim_matrix[i][j]
                
                # Jaccard Score
                jaccard_score = calculate_jaccard_similarity(processed_docs[i], processed_docs[j])
                
                # Classification
                verdict = classify_similarity(cosine_score, jaccard_score)
                
                results.append({
                    "File A": file1,
                    "File B": file2,
                    "Cosine Similarity": round(cosine_score, 4),
                    "Jaccard Similarity": round(jaccard_score, 4),
                    "Verdict": verdict
                })
        
        # Display Results using Pandas
        df_results = pd.DataFrame(results)
        
        # Sort by Cosine Similarity descending to show most suspicious first
        df_results = df_results.sort_values(by="Cosine Similarity", ascending=False)
        
        print("=" * 80)
        print("FINAL PLAGIARISM REPORT")
        print("=" * 80)
        print(df_results.to_string(index=False))
        print("\n" + "=" * 80)
        
        # Explanation of Results
        print("\nINTERPRETATION:")
        print("- High Similarity (> 0.8): Likely plagiarism or heavy copying.")
        print("- Moderate Similarity (0.4 - 0.8): Possible shared sources or paraphrasing.")
        print("- Low Similarity (< 0.4): Different content or topic.")
