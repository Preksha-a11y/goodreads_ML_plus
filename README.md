# Text-Based Prediction of Book Review Popularity
**Course**: UE24CS352A – Machine Learning Mini-Project  
**Problem Statement #41**: Text-Based Prediction of Book Review Popularity  
**Reference Paper**: Stanford CS229 (Spring 2021) – *Text-Based Prediction of Book Review Popularity* (Bridget Daly)  
**Team**: Preksha C Mudigoudar & Team  
**Dataset Source**: [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) (Goodreads Reviews & Metadata)

---

## 📌 Problem Overview
Predict whether an online Goodreads book review achieves viral community engagement ($\text{like\_share} > 0.02$) using reviewer metadata, linguistics, sentiment, and vocabulary tokens.

$$\text{like\_share} = \frac{\text{review\_likes} + \text{review\_comments}}{\sum_{\text{book}} (\text{review\_likes} + \text{review\_comments})}$$

---

## 🚀 Key Highlights & Novel Breakthroughs

1. **The Contextual Scale Breakthrough (ROC-AUC 0.842)**:
   - The Stanford reference paper hit a ceiling of ~0.700 ROC-AUC because it analyzed text in a vacuum without considering **book audience scale**.
   - By engineering relative social authority (`user_to_book_ratio`), controversy tension (`contrarian_score`), and structured effort (`review_effort`), feature correlation with popularity surged from $+0.22$ to $+0.47$, propelling model ROC-AUC to **0.842**.

2. **⭐ Imbalance Resolution: Cost-Sensitive Loss Weighting vs. Random Undersampling (Units 1 & 3)**:
   - Instead of discarding ~70% of majority-class training data via heuristic random undersampling, we formulate cost-sensitive learning on 100% of data using `scale_pos_weight = N_neg / N_pos ≈ 3.76` in Gradient Boosting and PyTorch (`BCEWithLogitsLoss`), and `class_weight='balanced'` in Logistic Regression.
   - Preserves 100% of training data while recovering **78.7% Sensitivity**.

3. **Stacking Ensemble Meta-Learner (Unit 3)**:
   - Blends Logistic Regression, HistGradientBoosting, and Random Forest base estimators using a meta-classifier, achieving **82.5% overall accuracy** and **93.7% Specificity**.

4. **🔬 Curricular Student Experiments**:
   - **Experiment 1 (Unit 2)**: ANN Architecture Depth (1-Layer vs 2-Layer ReviewMLP with Dropout vs 3-Layer) & Learning Rate sweeps.
   - **Experiment 2 (Unit 1)**: Empirical Decision Tree Bias-Variance Tradeoff curve ($d = 1 \dots 15$), verifying optimal generalization at $d=7$.

---

## 📂 Key Files & Deliverables

| File | Description |
|---|---|
| **`run_experiments.py`** | **Master benchmark runner** executing all baseline models, cost-sensitive add-ons, contextual scale models, and student ablations. |
| **`interactive_demo.py`** | **Instant interactive live demo (<0.1s startup)** using serialized `demo_model.pkl` for evaluation presentation. |
| **`generate_visual_analysis_plots.py`** | Generates all 4 publication-quality 300-DPI visual analysis figures. |
| **`Mini_Project_Report_UE24CS352A_Final.pdf`** | Official 4-page formal illustrated report with embedded figures and tables. |
| **`Mini_Project_Presentation_Deck.md`** | Complete 10-slide oral presentation script and deck. |
| `demo_model.pkl` | Serialized model, scaler, and contextual feature registry for instant demo startup. |

---

## 🛠️ How to Run Locally

### 1. Run Complete Benchmark Experiments
```powershell
python run_experiments.py
```
Outputs the complete master evaluation table, generates `roc_curves_comparison.png`, and computes the bias-variance curve `bias_variance_curve.png`.

### 2. Run Interactive Live Demo (For Presentation / Viva)
```powershell
python interactive_demo.py
```
- Starts up in **< 0.1s** using cached `demo_model.pkl`.
- Option 1: Tests a verified Viral / Popular Review (88.4% probability).
- Option 2: Tests a verified Buried Review (14.2% probability).
- Option 3: Lets the reviewer type custom text and metadata live!

### 3. Generate Visual Figures
```powershell
python generate_visual_analysis_plots.py
```
Generates all 4 visual analysis figures: `feature_distributions_comparison.png`, `why_log_transforms_matter.png`, `feature_importance_breakthrough.png`, and `confusion_matrices_breakthrough.png`.
