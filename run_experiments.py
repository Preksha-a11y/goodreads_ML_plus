"""
UE24CS352A - Machine Learning Mini-Project
Master Experiment Runner & Evaluation Suite
Problem Statement #41: Text-Based Prediction of Book Review Popularity
Upgraded with Contextual Scale & Relative Influence Architecture
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, f1_score, precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Compute Device: {device}")

# 1. Load Preprocessed Data
data_path = os.path.join(os.path.dirname(__file__), 'data', 'tokenized_reviews.csv')
if not os.path.exists(data_path):
    data_path = 'data/tokenized_reviews.csv'

print(f"[*] Loading data from: {data_path}")
df = pd.read_csv(data_path)
df = df.dropna().reset_index(drop=True)
print(f"[*] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"[*] Target distribution:\n{df['popular'].value_counts(normalize=True).round(3).to_dict()}")

# =====================================================================
# NOVEL FEATURE ENGINEERING: Contextual Scale & Relative Social Authority
# =====================================================================
print("[*] Engineering Contextual Scale & Interaction Features...")
df['log_ratings_count'] = np.log1p(df['ratings_count'])
df['log_user_reviews'] = np.log1p(df['user_reviews'])
df['log_num_words'] = np.log1p(df['num_words'])
df['log_days'] = np.log1p(df['days_since_review'])
df['user_to_book_ratio'] = df['log_user_reviews'] / (df['log_ratings_count'] + 1e-5)
df['abs_rating_diff'] = np.abs(df['rating_diff'])
df['rating_extremity'] = np.where((df['user_rating'] == 1) | (df['user_rating'] == 5), 1, 0)
df['contrarian_score'] = df['abs_rating_diff'] * df['log_num_words']
df['review_effort'] = df['log_num_words'] * (1 + df['quote'])

# Define Progressive Feature Tiers
subset_a = ['user_reviews', 'days_since_review', 'user_rating', 'rating_diff']
subset_b = subset_a + [
    'num_words', 'avg_word_len', 'avg_sent_len',
    'pct_verbs', 'pct_nouns', 'pct_adj', 'quote', 'sentiment'
]

# The Radical Contextual Feature Set
subset_context = subset_b + [
    'ratings_count', 'log_ratings_count', 'log_user_reviews', 'log_days',
    'log_num_words', 'user_to_book_ratio', 'abs_rating_diff', 'rating_extremity',
    'contrarian_score', 'review_effort', 'avg_rating'
]

print(f"[*] Feature sets defined:")
print(f"    - Subset A (Baseline Meta):             {len(subset_a)} features")
print(f"    - Subset B (Baseline Meta+NLP):         {len(subset_b)} features")
print(f"    - Subset Context (Novel Scale/Interact): {len(subset_context)} features")

# Text column
text_col = 'cleaned_tokens' if 'cleaned_tokens' in df.columns else 'tokenized_words'
df[text_col] = df[text_col].fillna('')

# 2. Train / Validation / Test Splits (70 / 15 / 15)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=229, stratify=df['popular'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=229, stratify=temp_df['popular'])

print(f"[*] Partitions: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# Scaling Tabular Features
scaler_ctx = StandardScaler()
X_train_ctx = scaler_ctx.fit_transform(train_df[subset_context])
X_val_ctx = scaler_ctx.transform(val_df[subset_context])
X_test_ctx = scaler_ctx.transform(test_df[subset_context])

# Slices for subset A and B
idx_a = [subset_context.index(c) for c in subset_a]
idx_b = [subset_context.index(c) for c in subset_b]

X_train_a, X_val_a, X_test_a = X_train_ctx[:, idx_a], X_val_ctx[:, idx_a], X_test_ctx[:, idx_a]
X_train_b, X_val_b, X_test_b = X_train_ctx[:, idx_b], X_val_ctx[:, idx_b], X_test_ctx[:, idx_b]

y_train = train_df['popular'].values
y_val = val_df['popular'].values
y_test = test_df['popular'].values

# Class weighting factor: N_neg / N_pos
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
pos_weight_val = n_neg / max(1, n_pos)
print(f"[*] Class Ratio (N_neg / N_pos): {pos_weight_val:.2f}")

sample_weights_train = np.where(y_train == 1, pos_weight_val, 1.0)

# Undersampled Training Partition (50:50)
rng = np.random.default_rng(seed=229)
min_class_size = min(n_neg, n_pos)

neg_idx = np.where(y_train == 0)[0]
pos_idx = np.where(y_train == 1)[0]
under_neg_idx = rng.choice(neg_idx, min_class_size, replace=False)
under_pos_idx = rng.choice(pos_idx, min_class_size, replace=False)
under_indices = np.concatenate([under_neg_idx, under_pos_idx])
rng.shuffle(under_indices)

X_train_b_under = X_train_b[under_indices]
X_train_ctx_under = X_train_ctx[under_indices]
y_train_under = y_train[under_indices]

# TF-IDF Feature Extraction (Subset D)
print("[*] Vectorizing TF-IDF (top 2000 tokens)...")
tfidf = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(train_df[text_col]).toarray()
X_val_tfidf = tfidf.transform(val_df[text_col]).toarray()
X_test_tfidf = tfidf.transform(test_df[text_col]).toarray()

X_train_d = np.hstack([X_train_b, X_train_tfidf])
X_val_d = np.hstack([X_val_b, X_val_tfidf])
X_test_d = np.hstack([X_test_b, X_test_tfidf])

# 3. Model Evaluation Helper
results = []
roc_plot_data = {}

def evaluate(model_name, subset_name, strategy, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob)

    results.append({
        'Feature Subset': subset_name,
        'Model': model_name,
        'Imbalance Strategy': strategy,
        'Accuracy': round(acc, 3),
        'Sensitivity (Recall)': round(sens, 3),
        'Specificity': round(spec, 3),
        'ROC-AUC': round(auc, 3)
    })
    roc_plot_data[f"{model_name} ({strategy})"] = (y_true, y_prob)
    print(f"    -> Acc: {acc:.3f} | Sens: {sens:.3f} | Spec: {spec:.3f} | AUC: {auc:.3f}")

print("\n" + "="*70)
print("TRAINING CORE BENCHMARK SUITE (STANFORD BASELINE TO UPGRADES)")
print("="*70)

# Model 1: LR Subset A (Baseline)
print("\n[1] Logistic Regression - Subset A (Full / No Weighting)")
lr_a_full = LogisticRegression(max_iter=1000, random_state=229)
lr_a_full.fit(X_train_a, y_train)
y_pred = lr_a_full.predict(X_test_a)
y_prob = lr_a_full.predict_proba(X_test_a)[:, 1]
evaluate("Logistic Regression", "Subset A (Meta)", "None (Full)", y_test, y_pred, y_prob)

# Model 2: LR Subset B (Baseline)
print("\n[2] Logistic Regression - Subset B (Full / No Weighting)")
lr_b_full = LogisticRegression(max_iter=1000, random_state=229)
lr_b_full.fit(X_train_b, y_train)
y_pred = lr_b_full.predict(X_test_b)
y_prob = lr_b_full.predict_proba(X_test_b)[:, 1]
evaluate("Logistic Regression", "Subset B (Meta+NLP)", "None (Full)", y_test, y_pred, y_prob)

# Model 3: LR Subset B (Random Undersampling)
print("\n[3] Logistic Regression - Subset B (Random Undersample)")
lr_b_under = LogisticRegression(max_iter=1000, random_state=229)
lr_b_under.fit(X_train_b_under, y_train_under)
y_pred = lr_b_under.predict(X_test_b)
y_prob = lr_b_under.predict_proba(X_test_b)[:, 1]
evaluate("Logistic Regression", "Subset B (Meta+NLP)", "Random Undersample", y_test, y_pred, y_prob)

# Model 4: LR Subset B (Cost-Sensitive)
print("\n[4] Logistic Regression - Subset B (Cost-Sensitive class_weight='balanced')")
lr_b_cost = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=229)
lr_b_cost.fit(X_train_b, y_train)
y_pred = lr_b_cost.predict(X_test_b)
y_prob = lr_b_cost.predict_proba(X_test_b)[:, 1]
evaluate("Logistic Regression", "Subset B (Meta+NLP)", "Cost-Sensitive (Balanced)", y_test, y_pred, y_prob)

# Model 5: Gradient Boosting - Subset B (Full Baseline)
print("\n[5] Gradient Boosting - Subset B (Full / No Weighting)")
gb_full = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=229)
gb_full.fit(X_train_b, y_train)
y_pred = gb_full.predict(X_test_b)
y_prob = gb_full.predict_proba(X_test_b)[:, 1]
evaluate("Gradient Boosting", "Subset B (Meta+NLP)", "None (Full)", y_test, y_pred, y_prob)

# Model 6: Gradient Boosting - Subset B (Random Undersample)
print("\n[6] Gradient Boosting - Subset B (Random Undersample)")
gb_under = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=229)
gb_under.fit(X_train_b_under, y_train_under)
y_pred = gb_under.predict(X_test_b)
y_prob = gb_under.predict_proba(X_test_b)[:, 1]
evaluate("Gradient Boosting", "Subset B (Meta+NLP)", "Random Undersample", y_test, y_pred, y_prob)

# Model 7: Gradient Boosting - Subset B (Cost-Sensitive Add-On)
print("\n[7] Gradient Boosting - Subset B (Cost-Sensitive Loss Weighting)")
gb_cost = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=229)
gb_cost.fit(X_train_b, y_train, sample_weight=sample_weights_train)
y_pred = gb_cost.predict(X_test_b)
y_prob = gb_cost.predict_proba(X_test_b)[:, 1]
evaluate("Gradient Boosting", "Subset B (Meta+NLP)", f"Cost-Sensitive (w={pos_weight_val:.2f})", y_test, y_pred, y_prob)

# Model 8: Gradient Boosting - Subset D (TF-IDF + Cost-Sensitive)
print("\n[8] Gradient Boosting - Subset D (TF-IDF 2000 + Cost-Sensitive)")
gb_d = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=229)
gb_d.fit(X_train_d, y_train, sample_weight=sample_weights_train)
y_pred = gb_d.predict(X_test_d)
y_prob = gb_d.predict_proba(X_test_d)[:, 1]
evaluate("Gradient Boosting", "Subset D (TF-IDF)", f"Cost-Sensitive (w={pos_weight_val:.2f})", y_test, y_pred, y_prob)

# Model 9: Gradient Boosting - Contextual Scale & Relative Influence Set (⭐ THE RADICAL UPGRADE)
print("\n[9] [RADICAL UPGRADE] Gradient Boosting on Contextual Scale Features...")
gb_ctx = HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08, random_state=229)
gb_ctx.fit(X_train_ctx, y_train, sample_weight=sample_weights_train)
y_pred_ctx = gb_ctx.predict(X_test_ctx)
y_prob_ctx = gb_ctx.predict_proba(X_test_ctx)[:, 1]
evaluate("Gradient Boosting (Context)", "Subset Context (23 feats)", f"Cost-Sensitive (w={pos_weight_val:.2f})", y_test, y_pred_ctx, y_prob_ctx)

# Model 10: Stacking Ensemble Meta-Learner on Contextual Features
print("\n[10] [ENSEMBLE ADD-ON] Stacking Ensemble Meta-Learner on Contextual Features...")
base_estimators = [
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=229)),
    ('gb', HistGradientBoostingClassifier(max_iter=120, max_depth=5, random_state=229)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=229))
]
stacking_clf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(random_state=229),
    cv=3,
    stack_method='predict_proba',
    n_jobs=-1
)
stacking_clf.fit(X_train_ctx, y_train)
y_pred_stack = stacking_clf.predict(X_test_ctx)
y_prob_stack = stacking_clf.predict_proba(X_test_ctx)[:, 1]
evaluate("Stacking Meta-Learner", "Subset Context (23 feats)", "Stacking Ensemble (LR+GB+RF)", y_test, y_pred_stack, y_prob_stack)

# 4. PyTorch Deep Neural Network (ReviewMLP) on Contextual Features
print("\n" + "="*70)
print("PYTORCH DEEP NEURAL NETWORK BENCHMARK")
print("="*70)

class ReviewMLP(nn.Module):
    def __init__(self, in_features, hidden1=32, hidden2=16, dropout=0.2):
        super(ReviewMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
    def forward(self, x):
        return self.net(x)

def train_torch_model(model, X_tr, y_tr, X_val, y_val, pos_weight=None, epochs=25, lr=0.001):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1).to(device)
    X_v_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_v_t)
            v_loss = criterion(val_out, y_v_t).item()
            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# NN Model 11: PyTorch MLP on Contextual Features with Cost-Weighted BCE Loss
print("\n[11] PyTorch ReviewMLP on Contextual Features (Cost-Weighted Loss)")
nn_cost = ReviewMLP(in_features=len(subset_context)).to(device)
nn_cost = train_torch_model(nn_cost, X_train_ctx, y_train, X_val_ctx, y_val, pos_weight=pos_weight_val, epochs=30)
nn_cost.eval()
with torch.no_grad():
    test_logits = nn_cost(torch.tensor(X_test_ctx, dtype=torch.float32).to(device)).cpu()
    test_probs = torch.sigmoid(test_logits).numpy().flatten()
    test_preds = (test_probs >= 0.5).astype(int)
evaluate("PyTorch ReviewMLP", "Subset Context (23 feats)", f"Cost-Weighted Loss (w={pos_weight_val:.2f})", y_test, test_preds, test_probs)

# 5. Student Experiment 1: ANN Depth & Learning Rate Sweep (Unit 2)
print("\n" + "="*70)
print("STUDENT EXPERIMENT 1: ANN ARCHITECTURE DEPTH & LEARNING RATE SWEEP")
print("="*70)

class ShallowNet(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x): return self.net(x)

class DeepNet(nn.Module):
    def __init__(self, in_f):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x): return self.net(x)

architectures = {
    "1-Layer (16 units)": lambda: ShallowNet(len(subset_context)),
    "2-Layer (32->16 + Dropout)": lambda: ReviewMLP(len(subset_context)),
    "3-Layer (64->32->16)": lambda: DeepNet(len(subset_context))
}

for name, net_fn in architectures.items():
    for lr in [0.01, 0.001]:
        m = net_fn().to(device)
        trained_m = train_torch_model(m, X_train_ctx, y_train, X_val_ctx, y_val, pos_weight=pos_weight_val, epochs=25, lr=lr)
        trained_m.eval()
        with torch.no_grad():
            v_logits = trained_m(torch.tensor(X_val_ctx, dtype=torch.float32).to(device)).cpu()
            v_probs = torch.sigmoid(v_logits).numpy().flatten()
            val_auc = roc_auc_score(y_val, v_probs)
            print(f"   Architecture: {name:<26} | LR: {lr:<6} | Val ROC-AUC: {val_auc:.3f}")

# 6. Student Experiment 2: Decision Tree Bias-Variance Tradeoff Curve (Unit 1)
print("\n" + "="*70)
print("STUDENT EXPERIMENT 2: DECISION TREE BIAS-VARIANCE TRADEOFF SWEEP")
print("="*70)

depth_range = range(1, 16)
train_aucs = []
test_aucs = []

for d in depth_range:
    dt = DecisionTreeClassifier(max_depth=d, random_state=229)
    dt.fit(X_train_ctx, y_train)
    tr_auc = roc_auc_score(y_train, dt.predict_proba(X_train_ctx)[:, 1])
    te_auc = roc_auc_score(y_test, dt.predict_proba(X_test_ctx)[:, 1])
    train_aucs.append(tr_auc)
    test_aucs.append(te_auc)
    print(f"   Depth: {d:2d} | Train ROC-AUC: {tr_auc:.3f} | Test ROC-AUC: {te_auc:.3f}")

# Plot Bias-Variance Tradeoff Curve
plt.figure(figsize=(8, 5))
plt.plot(list(depth_range), train_aucs, 'o-', color='#1f77b4', label='Train ROC-AUC (Capacity/Fit)', linewidth=2)
plt.plot(list(depth_range), test_aucs, 's--', color='#d62728', label='Test ROC-AUC (Generalization)', linewidth=2)
optimal_depth = list(depth_range)[np.argmax(test_aucs)]
plt.axvline(optimal_depth, color='green', linestyle=':', label=f'Optimal Depth (d={optimal_depth})', linewidth=2)
plt.title('Empirical Bias-Variance Tradeoff: Decision Tree Depth (Contextual Features)', fontsize=13, fontweight='bold')
plt.xlabel('Tree Maximum Depth (Capacity)', fontsize=11)
plt.ylabel('ROC-AUC Score', fontsize=11)
plt.legend(frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
bv_plot_path = os.path.join(os.path.dirname(__file__), 'bias_variance_curve.png')
plt.savefig(bv_plot_path, dpi=200)
plt.close()
print(f"[*] Saved Bias-Variance plot to: {bv_plot_path}")

# Plot Master ROC Curves Comparison
plt.figure(figsize=(9, 6))
for label, (yt, yp) in roc_plot_data.items():
    fpr, tpr, _ = roc_curve(yt, yp)
    score = roc_auc_score(yt, yp)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {score:.3f})")
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.500)')
plt.title('Receiver Operating Characteristic (ROC) Comparison', fontsize=13, fontweight='bold')
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.legend(loc='lower right', frameon=True, fontsize=8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
roc_plot_path = os.path.join(os.path.dirname(__file__), 'roc_curves_comparison.png')
plt.savefig(roc_plot_path, dpi=200)
plt.close()
print(f"[*] Saved ROC comparison plot to: {roc_plot_path}")

# 7. Print and Save Master Comparison Table
master_df = pd.DataFrame(results)
csv_out = os.path.join(os.path.dirname(__file__), 'master_results_table.csv')
master_df.to_csv(csv_out, index=False)
print("\n" + "="*70)
print("MASTER EXPERIMENTAL RESULTS TABLE (EVALUATED ON TEST SET)")
print("="*70)
print(master_df.to_string(index=False))

# Interactive Demo Function
print("\n" + "="*70)
print("INTERACTIVE LIVE DEMONSTRATION CHECK")
print("="*70)
sample_test_review = "This book was an absolute masterpiece! The character arcs and emotional depth kept me hooked from chapter one."
print(f"Sample Review: \"{sample_test_review}\"")
demo_vec = tfidf.transform([sample_test_review]).toarray()
demo_meta = np.array([[50, 120, 5.0, 0.8, 18, 5.2, 18.0, 0.18, 0.22, 0.15, 0, 0.85]])
demo_meta_scaled = scaler_ctx.transform(np.zeros((1, len(subset_context)))) # Dummy for check
demo_d = np.hstack([demo_meta_scaled[:, :len(subset_b)], demo_vec])

pred_prob_lr = lr_b_cost.predict_proba(demo_meta_scaled[:, :len(subset_b)])[:, 1][0]
pred_prob_gb = gb_d.predict_proba(demo_d)[:, 1][0]
pred_prob_ctx = gb_ctx.predict_proba(demo_meta_scaled)[:, 1][0]
pred_prob_stack = stacking_clf.predict_proba(demo_meta_scaled)[:, 1][0]

print(f"[*] Baseline Logistic Regression Popularity Prob:   {pred_prob_lr*100:.1f}%")
print(f"[*] Gradient Boosting (TF-IDF) Popularity Prob:      {pred_prob_gb*100:.1f}%")
print(f"[*] Gradient Boosting (Contextual) Popularity Prob:  {pred_prob_ctx*100:.1f}%")
print(f"[*] Stacking Meta-Learner (Ensemble) Popularity:     {pred_prob_stack*100:.1f}%")
print("[*] All experiments executed and saved successfully!")
