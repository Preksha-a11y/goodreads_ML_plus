"""
High-Performance Multi-Core NLP Feature Extractor for 100,000 Goodreads Reviews
Utilizes all available CPU cores for parallel NLTK POS-tagging and VADER sentiment scoring.
"""

import os
import sys
import time
import multiprocessing as mp
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK data available
for res in ['vader_lexicon', 'punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

# Worker initialization
_sid = None
_wnl = None
_stopwords = None

def init_worker():
    global _sid, _wnl, _stopwords
    _sid = SentimentIntensityAnalyzer()
    _wnl = WordNetLemmatizer()
    _stopwords = set(stopwords.words('english'))

def process_single_row(row_dict):
    text = str(row_dict.get('review_text', ''))
    user_rating = float(row_dict.get('user_rating', 0.0))
    avg_rating = float(row_dict.get('avg_rating', 0.0))
    
    rating_diff = user_rating - avg_rating
    quote = int('"' in text)
    
    # Sentence tokenization
    try:
        sents = nltk.sent_tokenize(text)
    except Exception:
        sents = [text]
    num_sentences = max(1, len(sents))
    
    # Word tokenization
    try:
        raw_words = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    except Exception:
        raw_words = [w.lower() for w in text.split() if w.isalpha()]
    num_words = max(1, len(raw_words))
    
    avg_sent_len = num_words / num_sentences
    avg_word_len = sum(len(w) for w in raw_words) / num_words
    
    # POS tagging
    try:
        pos_tags = nltk.pos_tag(raw_words)
    except Exception:
        pos_tags = [(w, 'NN') for w in raw_words]
        
    pct_verbs = sum(1 for w in pos_tags if w[1].startswith('V')) / num_words
    pct_nouns = sum(1 for w in pos_tags if w[1].startswith('N')) / num_words
    pct_adj = sum(1 for w in pos_tags if w[1].startswith('J') or w[1].startswith('R')) / num_words
    
    # Sentiment
    if sents:
        sentiment = sum(_sid.polarity_scores(s)['compound'] for s in sents) / len(sents)
    else:
        sentiment = 0.0
        
    # Lemmatization & token cleaning
    cleaned_tokens = " ".join([_wnl.lemmatize(w) for w in raw_words if w not in _stopwords])
    
    return {
        'book_id': str(row_dict.get('book_id', '')),
        'user_reviews': int(row_dict.get('user_reviews', 0)),
        'user_rating': user_rating,
        'avg_rating': avg_rating,
        'ratings_count': int(row_dict.get('ratings_count', 0)),
        'days_since_review': int(row_dict.get('days_since_review', 0)),
        'review_likes': int(row_dict.get('review_likes', 0)),
        'like_share': float(row_dict.get('like_share', 0.0)),
        'popular': int(row_dict.get('popular', 0)),
        'rating_diff': round(rating_diff, 3),
        'num_words': num_words,
        'avg_word_len': round(avg_word_len, 3),
        'avg_sent_len': round(avg_sent_len, 3),
        'pct_verbs': round(pct_verbs, 3),
        'pct_nouns': round(pct_nouns, 3),
        'pct_adj': round(pct_adj, 3),
        'quote': quote,
        'sentiment': round(sentiment, 3),
        'cleaned_tokens': cleaned_tokens
    }

def main():
    start_time = time.time()
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(base_dir, 'data', 'goodreads_processed_samples', 'filtered_reviews_100k.csv')
    output_csv = os.path.join(base_dir, 'data', 'tokenized_reviews_100k.csv')
    
    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} not found!")
        sys.exit(1)
        
    print(f"[*] Reading 100k reviews from: {input_csv}...")
    df = pd.read_csv(input_csv)
    total_rows = len(df)
    print(f"[*] Total rows loaded: {total_rows:,}")
    print(f"[*] Class distribution:\n{df['popular'].value_counts(normalize=True).round(3).to_dict()}")
    
    # Setup multiprocessing pool (using 10 cores)
    num_workers = min(10, max(1, os.cpu_count() - 2))
    print(f"[*] Launching parallel processing with {num_workers} CPU worker processes...")
    
    # Convert dataframe to list of dicts for lightweight IPC
    records = df.to_dict('records')
    del df
    
    chunk_size = 500
    results = []
    
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap(process_single_row, records, chunksize=chunk_size)):
            results.append(res)
            if (i + 1) % 10_000 == 0 or (i + 1) == total_rows:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                remaining = (total_rows - (i + 1)) / speed if speed > 0 else 0
                print(f"    Processed {i+1:,} / {total_rows:,} rows ({((i+1)/total_rows)*100:.1f}%) | Speed: {speed:.0f} rows/s | ETA: {remaining:.0f}s")
                
    print("[*] Assembling output dataframe...")
    out_df = pd.DataFrame(results)
    print(f"[*] Saving to: {output_csv}...")
    out_df.to_csv(output_csv, index=False)
    
    # Also update the default tokenized_reviews.csv so all scripts use this 100k version
    default_csv = os.path.join(base_dir, 'data', 'tokenized_reviews.csv')
    out_df.to_csv(default_csv, index=False)
    print(f"[*] Also updated default tokenized_reviews.csv ({os.path.getsize(default_csv)/(1024**2):.2f} MB)")
    
    total_time = time.time() - start_time
    print(f"[SUCCESS] Feature extraction completed in {total_time:.1f}s ({total_time/60:.2f} mins)!")

if __name__ == '__main__':
    mp.freeze_support()
    main()
