"""
Interactive Live Demo for Reviewer / Viva Presentation
Quickly loads precomputed features and evaluates custom review inputs in real-time.
"""

import os
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Ensure NLTK resources
for res in ['vader_lexicon', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

import pickle

def load_or_train_demo_model():
    model_pkl = os.path.join(os.path.dirname(__file__), 'demo_model.pkl')
    if os.path.exists(model_pkl):
        print("[*] Loading pre-trained model from demo_model.pkl (instant startup)...")
        with open(model_pkl, 'rb') as f:
            data = pickle.load(f)
        print("[*] Model loaded and ready for live queries!\n")
        return data['model'], data['scaler'], data['features']
    
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'tokenized_reviews.csv')
    print("[*] Loading training data for live demo...")
    df = pd.read_csv(data_path)
    
    # Contextual Feature Engineering
    df['log_ratings_count'] = np.log1p(df['ratings_count'])
    df['log_user_reviews'] = np.log1p(df['user_reviews'])
    df['log_num_words'] = np.log1p(df['num_words'])
    df['log_days'] = np.log1p(df['days_since_review'])
    df['user_to_book_ratio'] = df['log_user_reviews'] / (df['log_ratings_count'] + 1e-5)
    df['abs_rating_diff'] = np.abs(df['rating_diff'])
    df['rating_extremity'] = np.where((df['user_rating'] == 1) | (df['user_rating'] == 5), 1, 0)
    df['contrarian_score'] = df['abs_rating_diff'] * df['log_num_words']
    df['review_effort'] = df['log_num_words'] * (1 + df['quote'])
    
    features = [
        'user_reviews', 'days_since_review', 'user_rating', 'rating_diff',
        'num_words', 'avg_word_len', 'avg_sent_len', 'pct_verbs', 'pct_nouns', 'pct_adj', 'quote', 'sentiment',
        'ratings_count', 'log_ratings_count', 'log_user_reviews', 'log_days',
        'log_num_words', 'user_to_book_ratio', 'abs_rating_diff', 'rating_extremity',
        'contrarian_score', 'review_effort', 'avg_rating'
    ]
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['popular'].values
    
    w_pos = (y == 0).sum() / (y == 1).sum()
    sample_weights = np.where(y == 1, w_pos, 1.0)
    
    print("[*] Training Gradient Boosting model on Contextual Scale Features...")
    clf = HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, random_state=229)
    clf.fit(X, y, sample_weight=sample_weights)
    
    with open(model_pkl, 'wb') as f:
        pickle.dump({'model': clf, 'scaler': scaler, 'features': features}, f)
        
    print("[*] Model trained and ready for live queries!\n")
    return clf, scaler, features


def extract_features_from_text(text, user_reviews, ratings_count, user_rating, avg_rating, days_since_review):
    sid = SentimentIntensityAnalyzer()
    
    sents = nltk.sent_tokenize(text) if text else [text]
    num_sents = max(1, len(sents))
    words = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    num_words = max(1, len(words))
    
    avg_sent_len = num_words / num_sents
    avg_word_len = sum(len(w) for w in words) / num_words if words else 4.0
    
    try:
        pos_tags = nltk.pos_tag(words)
    except Exception:
        pos_tags = [(w, 'NN') for w in words]
        
    pct_verbs = sum(1 for w in pos_tags if w[1].startswith('V')) / num_words
    pct_nouns = sum(1 for w in pos_tags if w[1].startswith('N')) / num_words
    pct_adj = sum(1 for w in pos_tags if w[1].startswith('J') or w[1].startswith('R')) / num_words
    
    quote = int('"' in text)
    sentiment = sum(sid.polarity_scores(s)['compound'] for s in sents) / num_sents if sents else 0.0
    
    rating_diff = user_rating - avg_rating
    log_ratings_count = np.log1p(ratings_count)
    log_user_reviews = np.log1p(user_reviews)
    log_num_words = np.log1p(num_words)
    log_days = np.log1p(days_since_review)
    user_to_book_ratio = log_user_reviews / (log_ratings_count + 1e-5)
    abs_rating_diff = abs(rating_diff)
    rating_extremity = 1 if user_rating in [1.0, 5.0] else 0
    contrarian_score = abs_rating_diff * log_num_words
    review_effort = log_num_words * (1 + quote)
    
    feat_dict = {
        'user_reviews': user_reviews,
        'days_since_review': days_since_review,
        'user_rating': user_rating,
        'rating_diff': rating_diff,
        'num_words': num_words,
        'avg_word_len': avg_word_len,
        'avg_sent_len': avg_sent_len,
        'pct_verbs': pct_verbs,
        'pct_nouns': pct_nouns,
        'pct_adj': pct_adj,
        'quote': quote,
        'sentiment': sentiment,
        'ratings_count': ratings_count,
        'log_ratings_count': log_ratings_count,
        'log_user_reviews': log_user_reviews,
        'log_days': log_days,
        'log_num_words': log_num_words,
        'user_to_book_ratio': user_to_book_ratio,
        'abs_rating_diff': abs_rating_diff,
        'rating_extremity': rating_extremity,
        'contrarian_score': contrarian_score,
        'review_effort': review_effort,
        'avg_rating': avg_rating
    }
    return feat_dict

def run_interactive_demo():
    clf, scaler, feature_names = load_or_train_demo_model()
    
    print("="*75)
    print("  GOODREADS REVIEW POPULARITY PREDICTOR - INTERACTIVE LIVE DEMO")
    print("="*75)
    print("Preset Test Scenarios Available:")
    print("  [1] High-Engagement Contrarian Review (Indie book, high authority reviewer)")
    print("  [2] Casual Brief Review (Mega-bestseller, ordinary reader)")
    print("  [3] Custom Input (Type your own review live!)")
    print("  [4] Exit")
    
    while True:
        choice = input("\nSelect an option [1-4]: ").strip()
        
        if choice == '1':
            text = (
                '\"This is not just another fantasy novel; it is an unforgiving dissection of human ambition.\" '
                'While the community gave this book 4.6 stars, I found the pacing in chapters 4 through 8 entirely bloated. '
                'The character arcs, however, redeem the narrative with visceral emotional conflict.'
            )
            user_reviews = 450
            ratings_count = 1200
            user_rating = 2.0
            avg_rating = 4.6
            days = 150
            scenario_name = "High-Authority Contrarian Review on Niche/Indie Book"
            
        elif choice == '2':
            text = "Nice book! Really enjoyed the ending, loved the main characters."
            user_reviews = 8
            ratings_count = 850000
            user_rating = 4.0
            avg_rating = 4.2
            days = 30
            scenario_name = "Casual Short Review on Mega-Bestseller"
            
        elif choice == '3':
            print("\n--- Enter Custom Details ---")
            text = input("Review Text: ").strip()
            if not text:
                text = "Great read! Highly recommend to everyone."
            try:
                user_reviews = int(input("Reviewer Past Reviews Count [e.g. 50]: ") or "50")
                ratings_count = int(input("Book Total Ratings Count [e.g. 5000]: ") or "5000")
                user_rating = float(input("User Star Rating (1.0 - 5.0) [e.g. 5.0]: ") or "5.0")
                avg_rating = float(input("Book Average Rating (1.0 - 5.0) [e.g. 4.1]: ") or "4.1")
                days = int(input("Days Since Review Posted [e.g. 100]: ") or "100")
            except ValueError:
                print("Invalid numerical input, using defaults.")
                user_reviews, ratings_count, user_rating, avg_rating, days = 50, 5000, 5.0, 4.1, 100
            scenario_name = "Custom User Input Review"
            
        elif choice == '4':
            print("Exiting demo. Good luck with your viva presentation!")
            break
        else:
            print("Invalid choice, please select 1, 2, 3, or 4.")
            continue
            
        # Process and predict
        feats = extract_features_from_text(text, user_reviews, ratings_count, user_rating, avg_rating, days)
        row_df = pd.DataFrame([[feats[col] for col in feature_names]], columns=feature_names)
        row_scaled = scaler.transform(row_df)
        
        prob = clf.predict_proba(row_scaled)[:, 1][0]
        is_pop = prob >= 0.5
        
        print("\n" + "-"*75)
        print(f"SCENARIO: {scenario_name}")
        print(f"Sample Review: \"{text[:120]}...\"" if len(text) > 120 else f"Sample Review: \"{text}\"")
        print("\nKey Extracted Signal Metrics:")
        print(f"  • Review Length:          {feats['num_words']} words (Effort Score: {feats['review_effort']:.2f})")
        print(f"  • VADER Sentiment Score:  {feats['sentiment']:+.3f}")
        print(f"  • Contrarian Score:       {feats['contrarian_score']:.2f} (|Diff|: {feats['abs_rating_diff']:.1f} stars)")
        print(f"  • Relative Scale Ratio:   {feats['user_to_book_ratio']:.3f} (User Authority: {user_reviews} revs | Book Scale: {ratings_count:,} ratings)")
        print(f"  • Text Structure:         {feats['pct_nouns']*100:.1f}% nouns, {feats['pct_verbs']*100:.1f}% verbs, {feats['pct_adj']*100:.1f}% adjectives")
        
        print("\nMODEL VERDICT:")
        if is_pop:
            print(f"  >>> RESULT: POPULAR REVIEW (like_share > 2%) <<<")
            print(f"  Confidence Probability: {prob*100:.1f}% [HIGH ENGAGEMENT]")
            print("  Primary Driver: High relative influence ratio and extensive text/contrarian engagement.")
        else:
            print(f"  >>> RESULT: UNPOPULAR REVIEW (like_share <= 2%) <<<")
            print(f"  Confidence Probability: {prob*100:.1f}% [LOW ENGAGEMENT]")
            print("  Primary Driver: Low relative authority against book audience scale; review gets buried.")
        print("-" * 75)

if __name__ == '__main__':
    run_interactive_demo()
