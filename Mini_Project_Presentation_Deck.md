# Mini-Project Presentation & Review Session Deck
## UE24CS352A: Machine Learning (Mini-Project)
**Problem Statement #41**: Text-Based Prediction of Book Review Popularity  
**Reference Paper**: Stanford CS229 (Bridget Daly)  
**Team**: Preksha C Mudigoudar & Team  
**Evaluation Review Duration**: ~10–15 Minutes (Presentation + Live Demonstration + Q&A)

---

### Slide 1: Title & Overview
- **Title**: Text-Based Prediction of Book Review Popularity
- **Course**: UE24CS352A — Machine Learning Mini-Project
- **Team**: Preksha C Mudigoudar & Team
- **Core Mission**: Predict whether an online book review achieves viral community engagement ($\text{like\_share} > 0.02$) using metadata, text linguistics, sentiment, and vocabulary.
- **Key Breakthroughs**:
  1. Rigorous evaluation of **Cost-Sensitive Loss Weighting vs. Random Undersampling** (Units 1 & 3).
  2. **The Contextual Scale Breakthrough**: Engineering relative audience ratios and contrarian tension, shattering the Stanford paper's 0.700 ROC-AUC ceiling to **0.842**.
  3. Empirical student ablations: ANN Architecture/LR Sweep (Unit 2) and Decision Tree Bias-Variance Tradeoff Curves (Unit 1).

---

### Slide 2: Problem Statement & Motivation
- **The Engagement Skew**: On social reading platforms (Goodreads), review popularity follows a heavy power-law: ~79% of reviews receive virtually 0 likes/comments, while top reviews receive thousands.
- **Relative Like Share Metric**:
  $$\text{like\_share} = \frac{\text{review\_likes} + \text{review\_comments}}{\sum_{\text{book}} (\text{review\_likes} + \text{review\_comments})}$$
- **Binary Classification Target**:
  $$y = \begin{cases} 1 & \text{if } \text{like\_share} > 0.02 \quad (\text{Popular: 21\%}) \\ 0 & \text{otherwise} \quad (\text{Unpopular: 79\%}) \end{cases}$$
- **Why this Matters**: Helps recommendation engines surface authoritative reviews early rather than relying purely on time-accumulated vanity metrics.

---

### Slide 3: Dataset Ingestion & Preprocessing
- **Source**: UCSD Book Graph (Wan et al.) — 15.7M reviews and 2.3M books.
- **Streaming Pipeline**:
  1. Drop reviews with 0 likes/comments (removes uninformative inactive rows).
  2. Filter books with $< 10$ text reviews or $< 60$ total review likes.
  3. Language detection: Retain high-confidence English reviews.
  4. Temporal feature: $\text{days\_since\_review} = \max(\text{dates}) - \text{date\_added}$.
- **Data Splits**: 70% Train, 15% Validation, 15% Holdout Test (Stratified, Seed = 229).

---

### Slide 4: Feature Representation Progression
We evaluate model performance across 4 progressive feature tiers:
1. **Subset A (Metadata — 4 features)**:
   - Reviewer prominence (`user_reviews`), temporal recency (`days_since_review`), star rating (`user_rating`), and opinion discrepancy ($\text{rating\_diff} = \text{user\_rating} - \text{avg\_book\_rating}$).
2. **Subset B (Metadata + NLP & Sentiment — 12 features)**:
   - Subset A + word count (`num_words`), sentence length, word length, syntactic ratios (`pct_verbs`, `pct_nouns`, `pct_adj`), quotation flag (`quote`), and VADER compound sentiment.
3. **Subset D (Metadata + NLP + TF-IDF — 2,012 features)**:
   - Subset B + top 2,000 lemmatized TF-IDF tokens capturing review vocabulary.
4. **⭐ Subset Context (Contextual Scale & Relative Social Authority — 23 features)**:
   - Subset B + book audience scale (`ratings_count`, `log_ratings_count`), log transforms, `user_to_book_ratio`, `contrarian_score`, `review_effort`, and `rating_extremity`.

---

### Slide 5: The "Accuracy Paradox" & Imbalance Resolution
- **The Paradox**: Standard unweighted models trained on the raw imbalanced data achieve ~80% accuracy, but this is a statistical illusion:
  - Specificity = 98%, but **Sensitivity is catastrophic (6.5%–16.1%)**!
- **Baseline Fix (Stanford CS229)**: Random Undersampling to 50:50.
  - *Drawback*: Discards ~70% of majority class training observations, destroying valuable variance.
- **⭐ Our Add-On (Units 1 & 3)**: **Cost-Sensitive Loss Weighting**
  - Keeps 100% of data.
  - Logistic Regression: `class_weight='balanced'`.
  - Gradient Boosting: $\text{scale\_pos\_weight} = \frac{N_{\text{neg}}}{N_{\text{pos}}} \approx 3.76$.
  - PyTorch MLP: `BCEWithLogitsLoss(pos_weight=tensor([3.76]))`.

---

### Slide 6: Model Architecture & Implementation
1. **Logistic Regression**: SAGA solver with L2 regularization.
2. **Gradient Boosting (HistGradientBoosting / XGBoost)**: Fast histogram-based binning (`tree_method='hist'`), max depth 5–6, learning rate 0.08, n_estimators=150–200.
3. **Deep Neural Network (ReviewMLP in PyTorch)**:
   - Input(23) $\to$ Linear(32) $\to$ ReLU $\to$ Dropout(0.2) $\to$ Linear(16) $\to$ ReLU $\to$ Linear(1) $\to$ Sigmoid.
   - Cost-weighted `BCEWithLogitsLoss` with Adam optimizer ($\eta = 0.001$).
4. **Stacking Ensemble Meta-Learner (Unit 3)**:
   - Combines Logistic Regression, HistGradientBoosting, and Random Forest base learners with a Logistic Regression meta-classifier.

---

### Slide 7: Comprehensive Results Comparison

| Feature Subset | Model Architecture | Imbalance Strategy | Accuracy | Sensitivity | Specificity | ROC-AUC |
|---|---|---|:---:|:---:|:---:|:---:|
| Subset A (Meta) | Logistic Regression | Baseline (Unweighted) | 79.0% | 6.5% | 98.3% | 0.670 |
| Subset B (Meta+NLP) | Logistic Regression | Baseline (Unweighted) | 79.4% | 9.5% | 98.0% | 0.703 |
| Subset B (Meta+NLP) | Logistic Regression | Random Undersample | 69.2% | 55.7% | 72.7% | 0.705 |
| **Subset B (Meta+NLP)** | **Logistic Regression** | **Balanced Loss Weight** | **69.2%** | **55.6%** | **72.9%** | **0.706** |
| Subset B (Meta+NLP) | Gradient Boosting | Baseline (Unweighted) | 80.4% | 16.1% | 97.5% | 0.730 |
| Subset B (Meta+NLP) | Gradient Boosting | Random Undersample | 66.9% | 65.9% | 67.1% | 0.729 |
| **Subset B (Meta+NLP)** | **Gradient Boosting** | **Cost-Sensitive (w=3.76)** | **67.4%** | **64.9%** | **68.1%** | **0.730** |
| **Subset D (TF-IDF)** | **Gradient Boosting** | **TF-IDF 2000 (Cost-Sens)** | **68.1%** | **65.1%** | **68.9%** | **0.736** |
| **⭐ Subset Context (23f)** | **Gradient Boosting (Context)** | **Cost-Sensitive (w=3.76)** | **74.6%** | **78.7%** | **73.6%** | **0.842** |
| **⭐ Subset Context (23f)** | **Stacking Meta-Learner** | **Ensemble (LR+GB+RF)** | **82.5%** | 40.3% | **93.7%** | **0.841** |
| **⭐ Subset Context (23f)** | **PyTorch ReviewMLP** | **Cost-Weighted Loss (w=3.76)** | **74.2%** | **78.8%** | 73.0% | **0.841** |

> **Key Takeaways**:
> 1. Cost-Sensitive loss weighting resolves the imbalance without discarding ~36,000 majority reviews.
> 2. The Contextual Scale feature set shatters the 0.730 baseline ceiling, driving ROC-AUC straight to **0.842**.
> 3. The **Stacking Meta-Learner (Unit 3)** achieves the highest overall accuracy (**82.5%**) with a 93.7% Specificity rate.

---

### Slide 8: Student Ablations & Curricular Experiments
1. **Experiment 1 (Unit 2 — ANN Architecture Depth & Learning Rate Sweep)**:
   - Evaluated 1-Layer (16 units), 2-Layer (32$\to$16 + Dropout 0.2), and 3-Layer (64$\to$32$\to$16) on Contextual features under $\eta \in \{0.01, 0.001\}$.
   - **Finding**: All regularized variants converged stably to **0.835 – 0.836 Val ROC-AUC**. The 2-Layer network with 20% Dropout struck the optimal trade-off between computational efficiency and generalization.
2. **Experiment 2 (Unit 1 — Decision Tree Bias-Variance Tradeoff Curve)**:
   - Swept tree depth $d \in [1, 15]$ on Contextual features.
   - $d=1$ (Stump): High bias, underfitting (Train AUC: 0.710, Test AUC: 0.710).
   - $d=7$: **The Optimal Generalization Sweet Spot (Test AUC: 0.824)**.
   - $d \ge 12$: High variance, severe overfitting (Train AUC surged to **0.948**, Test AUC plummeted to **0.694**).
   - *Plot generated and saved*: `goodreads_ML_plus/bias_variance_curve.png`.

---

### Slide 9: Feature Attribution & The Contextual Breakthrough
- **Why Stanford CS229 Hit a Wall at 0.700**:
  - The paper analyzed text in a vacuum without considering **book audience scale**.
  - Reviewing a niche book with 50 ratings vs. a bestseller with 5,000,000 ratings has completely different visibility dynamics.
- **The Breakthrough Features**:
  1. `user_to_book_ratio`: Measures relative social authority ($\frac{\ln(1+\text{user\_reviews})}{\ln(1+\text{ratings\_count})}$).
  2. `contrarian_score`: Captures debate and controversy ($|\text{rating\_diff}| \times \ln(1+\text{num\_words})$).
  3. `review_effort`: Captures high-effort structured critiques ($\ln(1+\text{num\_words}) \times (1+\text{quote})$).
- **Result**: Feature correlation with review popularity jumped from **+0.22 to +0.47**.

---

### Slide 10: Conclusions & Live Interactive Demo
- **Conclusions**:
  1. Solved class imbalance using Cost-Sensitive Loss Weighting, preserving 100% of data.
  2. Engineered Contextual Scale representations, pushing ROC-AUC to **0.842**.
  3. Validated bias-variance tradeoff curves (Unit 1), neural architecture depth (Unit 2), and ensemble meta-learning (Unit 3).
- **Live Terminal Demonstration**:
  ```powershell
  cd goodreads_ML_plus
  python interactive_demo.py
  ```
  - Instant loading (<0.1s) via pre-computed `demo_model.pkl`.
  - Live testing of viral reviews (88.4% probability), buried reviews (14.2% probability), and custom real-time inputs.
