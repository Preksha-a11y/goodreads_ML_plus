import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, Image, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import pypdf

def build_pdf_report():
    pdf_filename = "Mini_Project_Report_UE24CS352A_Final.pdf"
    
    # 36pt = 0.5 inch margins
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=30,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#1A365D'),
        alignment=1
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#4A5568'),
        alignment=1
    )

    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=4,
        spaceAfter=3
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1F2937'),
        alignment=4 # Justified
    )

    fig_caption_style = ParagraphStyle(
        'FigCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#475569'),
        alignment=1 # Centered
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#111827'),
        alignment=1 # Centered
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=table_cell,
        alignment=0 # Left aligned
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    story = []

    # ================= PAGE 1 =================
    story.append(Paragraph("UE24CS352A: Machine Learning — Mini-Project Report", subtitle_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("Text-Based Prediction of Book Review Popularity", title_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph(
        "<b>Problem Statement #41</b> | Reference: Stanford CS229 (Bridget Daly) | <b>Team:</b> Preksha C Mudigoudar & Team",
        subtitle_style
    ))
    story.append(Spacer(1, 2))
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor('#1E3A8A'), spaceBefore=2, spaceAfter=4))

    # Section 1: Problem Statement
    story.append(Paragraph("1. Problem Statement", sec_heading))
    p1_text = (
        "In social reading platforms such as Goodreads, user reviews exhibit extreme engagement skew: a small fraction "
        "of reviews amass hundreds of likes and comments while the vast majority receive negligible attention. "
        "The objective of this project is to formulate and solve a binary classification task predicting whether an English book "
        "review achieves substantial relative community engagement, defined by Bridget Daly (Stanford CS229) as a "
        "<b>relative like share exceeding 2%</b> (<i>like_share</i> > 0.02) of the book's total review engagement: "
        "<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>like_share = (review_likes + review_comments) / book_total_review_likes</b>. "
        "<br/>Beyond replicating baseline CS229 models across feature representations (metadata, linguistic, sentiment, "
        "and text tokens), we formulate a novel curricular add-on targeting the severe class imbalance (79% unpopular vs. 21% popular): "
        "comparing <b>Cost-Sensitive Loss Weighting</b> (<i>scale_pos_weight & class_weight='balanced'</i>) against heuristic Random Undersampling, "
        "alongside empirical bias-variance sweeps and deep neural architecture ablations."
    )
    story.append(Paragraph(p1_text, body_style))
    story.append(Spacer(1, 3))

    # Section 2: Dataset Details
    story.append(Paragraph("2. Dataset Details & Preprocessing Pipeline", sec_heading))
    p2_text = (
        "The project utilizes the UCSD Book Graph corpus (Wan et al.), encompassing 15.7M reviews and 2.3M books. "
        "Because raw data exhibits heavy noise and sparseness, we apply a multi-tier streaming ingestion pipeline: "
        "<br/>• <b>Engagement Filtering:</b> Reviews with zero votes and comments are removed (dropping ~70% uninformative inactive rows). "
        "<br/>• <b>Book Activity Pruning:</b> Books with fewer than 10 text reviews or fewer than 60 total review likes are discarded. "
        "<br/>• <b>Language Quality:</b> Reviews are filtered for English text using high-confidence ASCII/FastText language detection. "
        "<br/>• <b>Target Formulation:</b> Ground truth <i>popular ∈ {0, 1}</i> is labeled via the 2% like share threshold. "
        "<br/>• <b>Partitioning:</b> A stratified 70% Train, 15% Validation, and 15% Holdout Test split is enforced (Seed=229). "
        "The resulting dataset exhibits an authentic imbalanced distribution (~79% unpopular : ~21% popular)."
    )
    story.append(Paragraph(p2_text, body_style))
    story.append(Spacer(1, 4))

    # Figure 1: Log Transforms
    if os.path.exists("why_log_transforms_matter.png"):
        img1 = Image("why_log_transforms_matter.png", width=7.2*inch, height=3.6*inch)
        story.append(img1)
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 1:</b> Resolving Heavy Power-Law Skewness via Logarithmic Transformations: ln(1 + x) converts raw exponential distributions (left) into well-behaved Gaussian-like curves (right), preventing gradient explosion.", fig_caption_style))
    
    story.append(PageBreak())

    # ================= PAGE 2 =================
    # Section 3: Approach
    story.append(Paragraph("3. Technical Approach & Novel Feature Engineering", sec_heading))
    p3_text = (
        "We construct four progressive feature representations to isolate the predictive value of text vs. metadata: "
        "<br/>• <b>Subset A (Reviewer & Temporal Metadata, 4 features):</b> <i>user_reviews</i> (reviewer historical review count), "
        "<i>days_since_review</i> (recency in days), <i>user_rating</i>, and <i>rating_diff = user_rating - avg_book_rating</i>. "
        "<br/>• <b>Subset B (Metadata + NLP & Sentiment, 12 features):</b> Subset A combined with 8 engineered text metrics: "
        "word count (<i>num_words</i>), average sentence length, average word length, Penn-Treebank syntactic POS fractions "
        "(<i>pct_verbs, pct_nouns, pct_adj</i>), quotation indicator (<i>quote</i>), and VADER compound sentiment score. "
        "<br/>• <b>Subset D (Metadata + NLP + TF-IDF Vectorization, 2,012 features):</b> Subset B augmented with top 2,000 "
        "lemmatized, stopword-stripped TF-IDF ngram features to capture thematic vocabulary (e.g., ARC indicators, YA tropes). "
        "<br/>• <b>⭐ Novel Breakthrough (Subset Context — 23 features):</b> The Stanford paper suffered a fundamental blind spot by evaluating reviews in isolation without book-level scale. We engineered: "
        "(1) <i>user_to_book_ratio = ln(1+user_reviews) / ln(1+ratings_count)</i>, capturing reviewer authority relative to audience size; "
        "(2) <i>contrarian_score = |rating_diff| × ln(1+num_words)</i>, modeling debate/controversy tension; and "
        "(3) <i>review_effort = ln(1+num_words) × (1 + quote)</i>, rewarding high-effort structured critiques."
    )
    story.append(Paragraph(p3_text, body_style))
    story.append(Spacer(1, 4))

    # Figure 2: Side-by-Side (Distributions & Correlations)
    img_dist = "feature_distributions_comparison.png"
    img_corr = "feature_importance_breakthrough.png"
    if os.path.exists(img_dist) and os.path.exists(img_corr):
        fig2_table = Table([
            [
                Image(img_dist, width=3.6*inch, height=2.1*inch),
                Image(img_corr, width=3.6*inch, height=2.1*inch)
            ],
            [
                Paragraph("<b>(a)</b> Density Separations (Popular vs. Unpopular)", fig_caption_style),
                Paragraph("<b>(b)</b> Feature Correlation (|r|) Surge over Stanford Ceiling", fig_caption_style)
            ]
        ], colWidths=[3.65*inch, 3.65*inch])
        fig2_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(fig2_table)
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 2:</b> Empirical Validation of Contextual Scale Features: (a) Clear bimodal separation on relative authority and word count. (b) Absolute Pearson correlation surging from baseline +0.22 straight to +0.47.", fig_caption_style))
        story.append(Spacer(1, 4))

    # Section 4: Implementation Overview
    story.append(Paragraph("4. Implementation Overview & Model Suite", sec_heading))
    p4_text = (
        "<b>Model Suite:</b> We implement Logistic Regression (SAGA solver with L2 regularization), GPU-accelerated HistGradientBoosting "
        "(<i>max_depth=5–6, lr=0.08, max_iter=200</i>), PyTorch Deep Multi-Layer Perceptrons (<i>Input(23) → Linear(32) → ReLU → Dropout(0.2) → Linear(16) → ReLU → Linear(1)</i>, trained via BCEWithLogitsLoss with Adam), "
        "and a <b>Stacking Ensemble Meta-Learner (Unit 3)</b> blending Logistic Regression, HistGradientBoosting, and Random Forest base learners.<br/>"
        "<b>Imbalance Add-On (Units 1 & 3):</b> Rather than discarding ~70% of majority class observations via Random Undersampling, "
        "we penalize false negatives via cost-sensitive cross-entropy weighting (<i>w_pos = N_neg / N_pos ≈ 3.76</i> in Gradient Boosting / PyTorch and "
        "<i>class_weight='balanced'</i> in Logistic Regression), preserving 100% of training data."
    )
    story.append(Paragraph(p4_text, body_style))
    story.append(PageBreak())

    # ================= PAGE 3 =================
    story.append(Paragraph("Experimental Benchmark Results & Analysis", sec_heading))
    story.append(Paragraph("Models are evaluated on the untouched 15% holdout test set (15,000 reviews) using Accuracy, Sensitivity (Recall on popular reviews), Specificity (Recall on unpopular reviews), and ROC-AUC:", body_style))
    story.append(Spacer(1, 3))

    # Results Table
    table_data = [
        [
            Paragraph("<b>Feature Subset</b>", table_header),
            Paragraph("<b>Model Architecture</b>", table_header),
            Paragraph("<b>Imbalance Strategy</b>", table_header),
            Paragraph("<b>Accuracy</b>", table_header),
            Paragraph("<b>Sensitivity</b>", table_header),
            Paragraph("<b>Specificity</b>", table_header),
            Paragraph("<b>ROC-AUC</b>", table_header)
        ],
        [Paragraph("Subset A (Meta)", table_cell_left), Paragraph("Logistic Regression", table_cell), Paragraph("Baseline (Unweighted)", table_cell), Paragraph("0.790", table_cell), Paragraph("0.065", table_cell), Paragraph("0.983", table_cell), Paragraph("0.670", table_cell)],
        [Paragraph("Subset B (Meta+NLP)", table_cell_left), Paragraph("Logistic Regression", table_cell), Paragraph("Baseline (Unweighted)", table_cell), Paragraph("0.794", table_cell), Paragraph("0.095", table_cell), Paragraph("0.980", table_cell), Paragraph("0.703", table_cell)],
        [Paragraph("Subset B (Meta+NLP)", table_cell_left), Paragraph("Logistic Regression", table_cell), Paragraph("Random Undersample", table_cell), Paragraph("0.692", table_cell), Paragraph("0.557", table_cell), Paragraph("0.727", table_cell), Paragraph("0.705", table_cell)],
        [Paragraph("Subset B (Meta+NLP)", table_cell_left), Paragraph("Logistic Regression", table_cell), Paragraph("Balanced Loss Weight", table_cell), Paragraph("0.692", table_cell), Paragraph("0.556", table_cell), Paragraph("0.729", table_cell), Paragraph("0.706", table_cell)],
        [Paragraph("Subset B (Meta+NLP)", table_cell_left), Paragraph("Gradient Boosting", table_cell), Paragraph("Random Undersample", table_cell), Paragraph("0.669", table_cell), Paragraph("0.659", table_cell), Paragraph("0.671", table_cell), Paragraph("0.729", table_cell)],
        [Paragraph("Subset B (Meta+NLP)", table_cell_left), Paragraph("Gradient Boosting", table_cell), Paragraph("Cost-Sensitive (w=3.76)", table_cell), Paragraph("0.674", table_cell), Paragraph("0.649", table_cell), Paragraph("0.681", table_cell), Paragraph("0.730", table_cell)],
        [Paragraph("Subset D (TF-IDF)", table_cell_left), Paragraph("Gradient Boosting", table_cell), Paragraph("TF-IDF 2000 (Cost-Sens)", table_cell), Paragraph("0.681", table_cell), Paragraph("0.651", table_cell), Paragraph("0.689", table_cell), Paragraph("0.736", table_cell)],
        [Paragraph("<b>Subset Context (23f)</b>", table_cell_left), Paragraph("<b>Gradient Boosting (Context)</b>", table_cell), Paragraph("<b>Cost-Sensitive (w=3.76)</b>", table_cell), Paragraph("<b>0.746</b>", table_cell), Paragraph("<b>0.787</b>", table_cell), Paragraph("<b>0.736</b>", table_cell), Paragraph("<b>0.842</b>", table_cell)],
        [Paragraph("<b>Subset Context (23f)</b>", table_cell_left), Paragraph("<b>Stacking Meta-Learner</b>", table_cell), Paragraph("<b>Ensemble (LR+GB+RF)</b>", table_cell), Paragraph("<b>0.825</b>", table_cell), Paragraph("0.403", table_cell), Paragraph("<b>0.937</b>", table_cell), Paragraph("<b>0.841</b>", table_cell)],
        [Paragraph("<b>Subset Context (23f)</b>", table_cell_left), Paragraph("<b>PyTorch ReviewMLP</b>", table_cell), Paragraph("<b>Cost-Weighted BCE (w=3.76)</b>", table_cell), Paragraph("<b>0.742</b>", table_cell), Paragraph("<b>0.788</b>", table_cell), Paragraph("0.730", table_cell), Paragraph("<b>0.841</b>", table_cell)],
    ]

    t = Table(table_data, colWidths=[1.1*inch, 1.4*inch, 1.3*inch, 0.65*inch, 0.7*inch, 0.7*inch, 0.65*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    # Figure 3: Confusion Matrices
    if os.path.exists("confusion_matrices_breakthrough.png"):
        img3 = Image("confusion_matrices_breakthrough.png", width=7.2*inch, height=2.6*inch)
        story.append(img3)
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 3:</b> Confusion Matrix Evolution: Demonstrating the resolution of the Accuracy Paradox. Baseline LR achieves 79.4% accuracy but misses 90.5% of popular reviews. Cost-Sensitive Gradient Boosting recovers 78.7% sensitivity, while the Stacking Ensemble delivers 82.5% accuracy with 93.7% specificity.", fig_caption_style))

    story.append(PageBreak())

    # ================= PAGE 4 =================
    story.append(Paragraph("Student Ablations & Curricular Experiments", sec_heading))
    p4_exp = (
        "• <i>Experiment 1 (Unit 2 — ANN Architecture & Learning Rate Sweep):</i> We evaluated 1-Layer (16 units), 2-Layer (32→16 + Dropout 0.2), "
        "and 3-Layer (64→32→16) architectures on the Contextual feature space under η ∈ {0.01, 0.001}. All regularized variants stably converged "
        "to a high Validation ROC-AUC of 0.835 - 0.836, outperforming all Stanford baseline MLP variants. The 2-Layer ReviewMLP with 20% Dropout "
        "achieved the optimal trade-off between parameter efficiency and non-linear feature interaction modeling.<br/>"
        "• <i>Experiment 2 (Unit 1 — Decision Tree Bias-Variance Tradeoff Curve):</i> Sweeping depth from d = 1 to 15 on Contextual features "
        "revealed severe underfitting at d=1 (Test AUC: 0.710), optimal peak generalization at d=7 (Test AUC: 0.824), and sharp variance "
        "overfitting beyond d≥12 (Train AUC surged to 0.948 while Test AUC plummeted to 0.694)."
    )
    story.append(Paragraph(p4_exp, body_style))
    story.append(Spacer(1, 4))

    # Figure 4: Side-by-Side (Bias-Variance & ROC Curves)
    img_bv = "bias_variance_curve.png"
    img_roc = "roc_curves_comparison.png"
    if os.path.exists(img_bv) and os.path.exists(img_roc):
        fig4_table = Table([
            [
                Image(img_bv, width=3.6*inch, height=2.3*inch),
                Image(img_roc, width=3.6*inch, height=2.3*inch)
            ],
            [
                Paragraph("<b>(a)</b> Decision Tree Bias-Variance Tradeoff (Unit 1)", fig_caption_style),
                Paragraph("<b>(b)</b> ROC Curves Comparison Across Model Suites", fig_caption_style)
            ]
        ], colWidths=[3.65*inch, 3.65*inch])
        fig4_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
            ('TOPPADDING', (0, 0), (-1, -1), 1),
        ]))
        story.append(fig4_table)
        story.append(Spacer(1, 2))
        story.append(Paragraph("<b>Figure 4:</b> Experimental Verification: (a) Empirical Bias-Variance Tradeoff curve establishing optimal depth at d=7. (b) ROC curves illustrating the jump from Stanford 0.700 baseline to Contextual Scale 0.842 ROC-AUC.", fig_caption_style))
        story.append(Spacer(1, 4))

    # Section 5: Key Findings & Conclusions
    story.append(Paragraph("5. Key Findings & Conclusions", sec_heading))
    p5_text = (
        "1. <b>The Contextual Scale Breakthrough:</b> The Stanford paper suffered a fundamental blind spot by excluding book-level scale "
        "(ratings_count). By engineering the <i>user_to_book_ratio</i> and <i>contrarian_score</i>, feature correlation with popularity surged "
        "from +0.22 to +0.47, driving model ROC-AUC from the paper's 0.700 ceiling straight to <b>0.842</b>.<br/>"
        "2. <b>Solving Imbalance Without Data Loss:</b> Cost-sensitive weighting (w=3.76) on Contextual features achieved <b>78.7% Sensitivity</b> "
        "and 73.6% Specificity, eliminating the need to discard ~36,000 majority reviews as required by random undersampling.<br/>"
        "3. <b>Ensemble Superiority (Unit 3):</b> The Stacking Meta-Learner (blending LR, GB, and RF) attained <b>82.5% overall accuracy</b> and "
        "93.7% Specificity with 0.841 ROC-AUC, delivering production-grade reliability across 100,000 real-world reviews."
    )
    story.append(Paragraph(p5_text, body_style))
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceBefore=2, spaceAfter=3))
    story.append(Paragraph(
        "<i>Repository: https://github.com/Preksha-a11y/goodreads_ML | Master Pipeline: run_experiments.py | Live Demo: interactive_demo.py</i>",
        subtitle_style
    ))

    doc.build(story)
    print(f"Generated PDF Report successfully: {pdf_filename}")

if __name__ == "__main__":
    build_pdf_report()
