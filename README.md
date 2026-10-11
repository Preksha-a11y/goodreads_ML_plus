# Text-Based Prediction of Book Review Popularity
**Course**: UE24CS352A – Machine Learning Mini-Project  
**Reference Paper**: Stanford CS229 (Spring 2021) – *Text-Based Prediction of Book Review Popularity* (Bridget Daly)  
**Dataset Source**: [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) (Goodreads Reviews & Metadata)

---

## 📌 Problem Overview
Predict whether a Goodreads book review achieves high relative community engagement ($\text{like\_share} > 0.02$) using reviewer metadata, linguistic/sentiment properties, and TF-IDF text features.

---

## 🚀 Key Highlights & Novel Add-Ons

1. **GPU-Accelerated Unified Kaggle Pipeline**:
   - Includes `goodreads_kaggle_pipeline.ipynb` which runs end-to-end in **< 15 minutes** using Kaggle GPU (PyTorch `.to('cuda')` and XGBoost `tree_method='hist', device='cuda'`).
2. **⭐ Official Add-On: Cost-Sensitive Loss Weighting vs. Random Undersampling (Units 1 & 3)**:
   - Instead of discarding ~70% of majority-class training data via random undersampling, we formulate cost-sensitive learning on 100% of data using `scale_pos_weight = N_neg / N_pos ≈ 3.76` in XGBoost and `class_weight='balanced'` in Logistic Regression.
   - Preserves dataset variance and achieves higher Specificity and ROC-AUC.
3. **🔬 Student Experimentation & Ablations**:
   - **Experiment 1 (Unit 2)**: ANN Architecture Depth (1-Layer vs 2-Layer with Dropout vs 3-Layer) & Learning Rate sweeps ($\eta \in \{0.01, 0.001\}$).
   - **Experiment 2 (Unit 1)**: Decision Tree Depth vs. Empirical Bias-Variance Tradeoff curve ($d = 1 \dots 15$).

---

## 📂 Repository Contents

| File | Description |
|---|---|
| **`goodreads_kaggle_pipeline.ipynb`** | **Master self-contained Kaggle Notebook with GPU acceleration, all models, visualizations, and live demo.** |
| `data_prep.py` | Filters raw Goodreads JSON datasets and generates `popular` binary target. |
| `features.py` | Extracts linguistic, syntactic, sentiment (VADER), and word features. |
| `save_undersample.py` | Generates 70/15/15 splits and 50:50 undersampled training partitions. |
| `logistic_ab.py` & `logistic_cd.py` | Logistic Regression baseline benchmarks. |
| `gb.py` & `gb_under.py` | Gradient Boosting (XGBoost) baseline benchmarks. |
| `neuralnet.py` | Deep Neural Network MLP benchmarks with SHAP values. |
| `explore_predictions.ipynb` | Exploratory prediction analysis. |

---

## 🛠️ How to Run on Kaggle (3 Simple Steps)

1. **Create a New Notebook on Kaggle**:
   - Attach the Goodreads datasets:
     - `goodreads-reviews-dedup-json`
     - `goodreads-books-json`
2. **Upload `goodreads_kaggle_pipeline.ipynb`**:
   - In the Kaggle notebook editor, go to **File -> Import Notebook** and select `goodreads_kaggle_pipeline.ipynb`.
3. **Turn on GPU Acceleration**:
   - In the right-hand panel under **Settings -> Accelerator**, choose **GPU T4 x2** (or GPU P100).
   - Click **Run All**. The entire pipeline, metrics table, ROC plots, and interactive predictor will complete in **~12 to 15 minutes**.
