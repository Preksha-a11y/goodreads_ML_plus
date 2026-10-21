import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        # Header rule
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(36, 11 * inch - 26, 8.5 * inch - 36, 11 * inch - 26)
        self.drawString(36, 11 * inch - 22, "UE24CS352A Machine Learning | Comprehensive Project Overview & Viva Defense Guide")
        
        # Footer rule
        self.line(36, 32, 8.5 * inch - 36, 32)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 36, 20, footer_text)
        self.drawString(36, 20, "Text-Based Prediction of Book Review Popularity (Problem Statement #41)")
        self.restoreState()

def create_guide_pdf():
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Project_Overview_And_Viva_Guide.pdf")
    
    # 36pt = 0.5 inch margins
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Typography styles
    doc_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0F172A'),
        alignment=0,
        spaceAfter=3
    )

    doc_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#475569'),
        spaceAfter=6
    )

    sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=3
    )

    subsec_title = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=5,
        spaceAfter=2
    )

    body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=3
    )

    body_tight = ParagraphStyle(
        'BodyTight',
        parent=body,
        fontSize=8,
        leading=10.5,
        spaceAfter=2
    )

    callout_box = ParagraphStyle(
        'Callout',
        parent=body,
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E3A8A')
    )

    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F172A'),
        alignment=1
    )

    td_left = ParagraphStyle(
        'TDLeft',
        parent=td_style,
        alignment=0
    )

    story = []

    # ================= TITLE & BANNER =================
    story.append(Paragraph("Text-Based Prediction of Book Review Popularity", doc_title))
    story.append(Paragraph(
        "<b>Complete Project Architecture, Feature Engineering, Hyperparameter Rationale & Viva Defense Guide</b><br/>"
        "Course: UE24CS352A Machine Learning | Problem Statement #41 | Reference: Stanford CS229 (Bridget Daly) | Team: Preksha C Mudigoudar & Team",
        doc_subtitle
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=2, spaceAfter=6))

    # ================= SECTION 1: THE BIG PICTURE =================
    story.append(Paragraph("1. The Core Problem & The 30-Second Elevator Pitch", sec_title))
    story.append(Paragraph(
        "• <b>The Platform & Phenomenon:</b> On Goodreads, book review engagement follows an extreme power-law distribution. "
        "Approximately 79% of reviews receive virtually zero attention, while the top ~21% capture the overwhelming majority of community discussion and likes.<br/>"
        "• <b>The Machine Learning Objective:</b> Predict whether a newly posted book review will achieve substantial relative community engagement based on the review's text linguistics, reviewer reputation, and book contextual scale.<br/>"
        "• <b>Mathematical Target Formulation:</b> Rather than using raw vanity like counts (which heavily penalize small indie books), engagement is measured relative to the book's total review activity:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>like_share</b> = (review_likes + review_comments) / book_total_review_likes<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;If <b>like_share &gt; 0.02 (2%)</b>, the review is classified as <b>Popular (y = 1)</b>; otherwise <b>Unpopular (y = 0)</b>.<br/>"
        "• <b>The 30-Second Viva Summary:</b> <i>\"We replicated the Stanford CS229 benchmark on 100,000 Goodreads reviews, resolved the 79:21 class imbalance via Cost-Sensitive Loss Weighting (recovering 78.7% Sensitivity without discarding data), and engineered novel Contextual Scale interaction features that shattered the published 0.700 ROC-AUC ceiling to 0.842.\"</i>",
        body
    ))
    story.append(Spacer(1, 3))

    # ================= SECTION 2: STANFORD PAPER DEFICIENCIES =================
    story.append(Paragraph("2. Stanford Baseline (Bridget Daly, 2021) vs. Why It Hit a Wall at 0.700", sec_title))
    story.append(Paragraph(
        "• <b>Baseline Dataset:</b> UCSD Book Graph (Wan et al.) — filtered down to 1.98M reviews across 54k books.<br/>"
        "• <b>Baseline Feature Tiers:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Subset A (Metadata — 4 features):</i> user_reviews, days_since_review, user_rating, rating_diff.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Subset B (Metadata + NLP — 12 features):</i> Subset A + word count, sentence length, word length, Penn-Treebank POS fractions (verbs, nouns, adjectives), quotation flag, and VADER compound sentiment.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <i>Subsets C & D (Vocabulary — 10,012 features):</i> Subset B + top 10,000 Bag-of-Words (C) or TF-IDF tokens (D).<br/>"
        "• <b>Why the Paper Hit a Performance Ceiling (ROC-AUC ≈ 0.700):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;1. <b>The Contextual Scale Blind Spot:</b> The Stanford author analyzed text and reviewer counts in a vacuum, completely omitting <b>book-level audience scale</b> (<i>ratings_count</i>). Having 50 past reviews on an indie book makes you a dominant influencer; having 50 reviews on Harry Potter leaves you completely invisible.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;2. <b>Destructive Random Undersampling:</b> The paper discarded ~70% of majority class training observations to force a 50:50 ratio, throwing away valuable variance and real-world boundary information.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;3. <b>Extreme Skewness:</b> The paper fed raw counts into linear models without addressing exponential power-law distributions.",
        body
    ))
    story.append(Spacer(1, 3))

    # ================= SECTION 3: THE 23 CONTEXTUAL FEATURES =================
    story.append(Paragraph("3. Our Novel Feature Engineering Suite: The 23 Contextual Features", sec_title))
    story.append(Paragraph(
        "To break the 0.700 ceiling, we engineered **Subset Context (23 features)**, transforming Pearson correlation with popularity from +0.22 to +0.47:",
        body
    ))
    
    features_table_data = [
        [
            Paragraph("<b>Feature Name</b>", th_style),
            Paragraph("<b>Category</b>", th_style),
            Paragraph("<b>Mathematical Formula / Definition</b>", th_style),
            Paragraph("<b>Predictive Signal & Real-World Intuition</b>", th_style)
        ],
        [
            Paragraph("<b>user_to_book_ratio</b>", td_left),
            Paragraph("Relative Scale", td_style),
            Paragraph("ln(1 + user_reviews) / [ln(1 + ratings_count) + 1e-5]", td_left),
            Paragraph("<b>Single strongest feature (|r| = 0.472).</b> Measures user prominence relative to book audience size. Ratio > 0.8 indicates dominant platform voice.", td_left)
        ],
        [
            Paragraph("<b>contrarian_score</b>", td_left),
            Paragraph("Controversy Tension", td_style),
            Paragraph("|user_rating - avg_rating| × ln(1 + num_words)", td_left),
            Paragraph("Captures debate tension (|r| = 0.385). Long reviews sharply disagreeing with community consensus generate heavy comments and viral upvotes.", td_left)
        ],
        [
            Paragraph("<b>review_effort</b>", td_left),
            Paragraph("Text Structure", td_style),
            Paragraph("ln(1 + num_words) × (1 + quote)", td_left),
            Paragraph("Rewards comprehensive critiques containing direct book quotations. High effort strongly correlates with community trust and saves.", td_left)
        ],
        [
            Paragraph("<b>rating_extremity</b>", td_left),
            Paragraph("Psychological", td_style),
            Paragraph("1 if user_rating ∈ {1.0, 5.0} else 0", td_left),
            Paragraph("Extreme praise (5★) or harsh criticism (1★) polarizes readers and drives significantly higher engagement than moderate 3★ reviews.", td_left)
        ],
        [
            Paragraph("<b>log_ratings_count</b>", td_left),
            Paragraph("Audience Scale", td_style),
            Paragraph("ln(1 + ratings_count)", td_left),
            Paragraph("Compresses book audience scale from 0–4,000,000 into a smooth 0–16 Gaussian curve, eliminating massive power-law outlier distortion.", td_left)
        ],
        [
            Paragraph("<b>log_user_reviews</b>", td_left),
            Paragraph("Authority Scale", td_style),
            Paragraph("ln(1 + user_reviews)", td_left),
            Paragraph("Reflects diminishing returns: jumping from 1 to 20 reviews provides huge authority gains; jumping from 2,000 to 2,020 is negligible.", td_left)
        ],
        [
            Paragraph("<b>log_num_words</b>", td_left),
            Paragraph("Linguistic Scale", td_style),
            Paragraph("ln(1 + num_words)", td_left),
            Paragraph("Normalizes review length. Distinguishes 10-word one-liners from 300-word essays without letting 2,000-word outliers skew model weights.", td_left)
        ],
        [
            Paragraph("<b>log_days</b>", td_left),
            Paragraph("Temporal", td_style),
            Paragraph("ln(1 + days_since_review)", td_left),
            Paragraph("Models visibility accumulation: 10 days early at launch week matters enormously; 10 days difference after 3 years is completely irrelevant.", td_left)
        ],
        [
            Paragraph("<b>abs_rating_diff</b>", td_left),
            Paragraph("Sentiment Gap", td_style),
            Paragraph("|user_rating - avg_rating|", td_left),
            Paragraph("Magnitude of opinion divergence from the community average regardless of positive/negative direction.", td_left)
        ],
        [
            Paragraph("<b>sentiment (VADER)</b>", td_left),
            Paragraph("NLP Sentiment", td_style),
            Paragraph("VADER Compound polarity score [-1.0, +1.0]", td_left),
            Paragraph("Captures emotional intensity. Passionate reviews (highly positive or critical) outperform neutral, lukewarm reviews.", td_left)
        ],
        [
            Paragraph("<b>pct_verbs, nouns, adj</b>", td_left),
            Paragraph("Penn-Treebank POS", td_style),
            Paragraph("NLTK POS Tag ratios / total words", td_left),
            Paragraph("Measures syntactic composition and descriptive richness. High adjective/noun density signals informative critique.", td_left)
        ],
    ]

    ft_table = Table(features_table_data, colWidths=[1.3*inch, 1.0*inch, 2.3*inch, 2.8*inch])
    ft_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F766E')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(ft_table)
    story.append(Spacer(1, 4))

    # Why Log Transforms Callout
    why_log_data = [[
        Paragraph("<b>★ Why do we take the natural logarithm ln(1 + x) everywhere? (Crucial Viva Question)</b><br/>"
                  "1. <b>Resolves Extreme Skewness:</b> Web engagement metrics follow exponential power laws. Raw ratings_count ranges from 5 to 4,000,000+ (a 800,000x gap!). Without log transforms, 99.2% of data is crushed into the first histogram bin, causing gradient explosions in neural networks and extreme leverage in linear models.<br/>"
                  "2. <b>Models Real Human Diminishing Returns:</b> A 10-day difference during book launch week is massive; a 10-day difference after 3 years is zero. Log transforms mathematically replicate this human cognitive reality.<br/>"
                  "3. <b>Enables Stable Ratios:</b> Taking ratios of logs creates smooth, normalized indicators (like user_to_book_ratio from 0.0 to 1.2).",
                  callout_box)
    ]]
    why_log_table = Table(why_log_data, colWidths=[7.4 * inch])
    why_log_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDF4')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#16A34A')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(why_log_table)
    story.append(PageBreak())

    # ================= SECTION 4: MODELS & HYPERPARAMETER RATIONALE =================
    story.append(Paragraph("4. Implemented Models & Rigorous Hyperparameter Choices", sec_title))
    story.append(Paragraph(
        "Every single model and hyperparameter in our codebase was selected based on computational constraints, mathematical theory, and syllabus alignment (Units 1, 2, and 3):",
        body
    ))

    models_data = [
        [
            Paragraph("<b>Model Architecture</b>", th_style),
            Paragraph("<b>Algorithm & Library</b>", th_style),
            Paragraph("<b>Exact Hyperparameters Used</b>", th_style),
            Paragraph("<b>Engineering Justification & Syllabus Connection</b>", th_style)
        ],
        [
            Paragraph("<b>Logistic Regression</b><br/>(Baselines & Balanced)", td_left),
            Paragraph("sklearn.linear_model<br/>LogisticRegression", td_style),
            Paragraph("• solver='saga'<br/>• penalty='l2'<br/>• C=1.0<br/>• max_iter=1000<br/>• class_weight='balanced'", td_left),
            Paragraph("• <b>Why SAGA?</b> Stochastic Average Gradient with Accelerated extensions. Unlike standard SGD which is noisy, or Batch GD which is slow on 100k rows, SAGA stores past gradients to achieve linear convergence on large, sparse matrices.<br/>• <b>Why L2 Penalty?</b> Penalizes sum of squared weights, preventing individual rare words from receiving massive overfitted coefficients.<br/>• <b>Why Balanced Weight?</b> Penalizes minority errors by N_neg / N_pos ≈ 3.76, shifting the linear decision threshold to recover sensitivity.", td_left)
        ],
        [
            Paragraph("<b>HistGradientBoosting</b><br/>(Contextual Breakthrough Champion)", td_left),
            Paragraph("sklearn.ensemble<br/>HistGradientBoosting<br/>Classifier", td_style),
            Paragraph("• max_iter=150–200<br/>• max_depth=5–6<br/>• learning_rate=0.08<br/>• sample_weight=3.76<br/>• random_state=229", td_left),
            Paragraph("• <b>Why Hist (Histogram-based)?</b> Bins continuous features into 256 discrete integer bins. Reduces split finding from O(N log N) to O(N), running 20x–50x faster with zero loss in mathematical precision.<br/>• <b>Why max_depth=5–6?</b> Student Experiment 2 proved single trees peak in generalization at d=7. Constraining boosting base trees to depth 5–6 ensures shallow weak learners that combine smoothly without overfitting.<br/>• <b>Why learning_rate=0.08?</b> Shrinkage factor. A conservative learning rate prevents overshooting loss minima across 150+ sequential trees.", td_left)
        ],
        [
            Paragraph("<b>Stacking Ensemble Meta-Learner</b><br/>(Production High-Specificity)", td_left),
            Paragraph("sklearn.ensemble<br/>StackingClassifier<br/>(Unit 3 Syllabus)", td_style),
            Paragraph("• Level 1 Base Learners:<br/>  - LR (class_weight='balanced')<br/>  - HistGB (depth=5, iter=120)<br/>  - Random Forest (100 trees, depth=8)<br/>• Level 2 Meta-Classifier:<br/>  - LogisticRegression()<br/>• cv=3, stack_method='predict_proba'", td_left),
            Paragraph("• <b>Why Stacking?</b> Fulfills Unit 3 (Ensemble Learning). Combines three fundamentally diverse hypothesis spaces: linear hyperplanes (LR), gradient-boosted residuals (GB), and bootstrap aggregation (Random Forest).<br/>• <b>Why LR as Meta-Learner?</b> Learns optimal calibrated weights for base model probabilities without introducing high non-linear variance.<br/>• <b>Why cv=3?</b> Uses 3-fold cross-validation out-of-fold predictions to prevent severe data leakage between base estimators and meta-learner.<br/>• <b>Result:</b> Attains highest overall accuracy (82.5%) and 93.7% Specificity.", td_left)
        ],
        [
            Paragraph("<b>PyTorch ReviewMLP</b><br/>(Deep Feedforward Neural Net)", td_left),
            Paragraph("torch.nn & torch.optim<br/>Custom ReviewMLP<br/>(Unit 2 Syllabus)", td_style),
            Paragraph("• Input(23) → Linear(32) → ReLU → Dropout(0.2) → Linear(16) → ReLU → Linear(1)<br/>• Loss: BCEWithLogitsLoss(pos_weight=3.76)<br/>• Optimizer: Adam(lr=0.001)<br/>• batch_size=256, epochs=30", td_left),
            Paragraph("• <b>Why 32→16 Architecture?</b> Student Experiment 1 proved a 2-layer MLP matches or exceeds a 3-layer net (0.836 Val AUC) while training in half the time with far fewer parameters.<br/>• <b>Why Dropout(0.2)?</b> Randomly deactivates 20% of hidden neurons during each forward pass, breaking feature co-adaptation and preventing overfitting.<br/>• <b>Why BCEWithLogitsLoss?</b> Combines Sigmoid activation and Binary Cross-Entropy into a single numerically stable formula using the log-sum-exp trick, preventing float underflow/overflow (ln(0) crashes) while cost-weighting positive samples.<br/>• <b>Why Adam (lr=0.001)?</b> Adapts individual learning rates using first and second gradient moments, ensuring stable convergence across heterogeneous tabular scales.", td_left)
        ],
    ]

    models_table = Table(models_data, colWidths=[1.3*inch, 1.1*inch, 2.2*inch, 2.8*inch])
    models_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(models_table)
    story.append(Spacer(1, 4))

    # Random Forest Construction Box
    rf_data = [[
        Paragraph("<b>★ How is Random Forest Built Inside the Stacking Classifier? (Unit 3 Core Topic)</b><br/>"
                  "1. <b>Row Bagging (Bootstrap Aggregation):</b> For 100 trees, 100 distinct training sets of size N are sampled <i>with replacement</i> from the 70,000 training reviews. Each tree sees a slightly different permutation of samples.<br/>"
                  "2. <b>Column Bagging (Random Feature Subsets):</b> At every single split node, the tree is restricted to evaluating only a random subset of features (√23 ≈ 5 features). This prevents dominant features (like user_to_book_ratio) from appearing at the root of every tree, forcing different trees to learn diverse patterns.<br/>"
                  "3. <b>Ensemble Averaging:</b> Individual tree predictions are averaged, reducing model variance by 1/√M while maintaining low bias.",
                  callout_box)
    ]]
    rf_table = Table(rf_data, colWidths=[7.4 * inch])
    rf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EFF6FF')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#3B82F6')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(rf_table)
    story.append(PageBreak())

    # ================= SECTION 5: MASTER BENCHMARK RESULTS =================
    story.append(Paragraph("5. Empirical Benchmark Results & The Accuracy Paradox", sec_title))
    story.append(Paragraph(
        "All models evaluated on the untouched 15% holdout test partition (15,000 reviews: 11,850 unpopular, 3,150 popular). "
        "Ground truth results saved in <i>goodreads_ML_plus/master_results_table.csv</i>:",
        body
    ))

    table_data = [
        [
            Paragraph("<b>Feature Subset</b>", th_style),
            Paragraph("<b>Model Architecture</b>", th_style),
            Paragraph("<b>Imbalance Strategy</b>", th_style),
            Paragraph("<b>Accuracy</b>", th_style),
            Paragraph("<b>Sensitivity</b>", th_style),
            Paragraph("<b>Specificity</b>", th_style),
            Paragraph("<b>ROC-AUC</b>", th_style)
        ],
        [Paragraph("Subset A (Meta)", td_left), Paragraph("Logistic Regression", td_style), Paragraph("Baseline (Unweighted)", td_style), Paragraph("79.0%", td_style), Paragraph("6.5%", td_style), Paragraph("98.3%", td_style), Paragraph("0.670", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("Logistic Regression", td_style), Paragraph("Baseline (Unweighted)", td_style), Paragraph("79.4%", td_style), Paragraph("9.5%", td_style), Paragraph("98.0%", td_style), Paragraph("0.703", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("Logistic Regression", td_style), Paragraph("Random Undersample", td_style), Paragraph("69.2%", td_style), Paragraph("55.7%", td_style), Paragraph("72.7%", td_style), Paragraph("0.705", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("<b>Logistic Regression</b>", td_style), Paragraph("<b>Balanced Loss Weight</b>", td_style), Paragraph("69.2%", td_style), Paragraph("55.6%", td_style), Paragraph("72.9%", td_style), Paragraph("<b>0.706</b>", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("Gradient Boosting", td_style), Paragraph("Baseline (Unweighted)", td_style), Paragraph("80.4%", td_style), Paragraph("16.1%", td_style), Paragraph("97.5%", td_style), Paragraph("0.730", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("Gradient Boosting", td_style), Paragraph("Random Undersample", td_style), Paragraph("66.9%", td_style), Paragraph("65.9%", td_style), Paragraph("67.1%", td_style), Paragraph("0.729", td_style)],
        [Paragraph("Subset B (Meta+NLP)", td_left), Paragraph("<b>Gradient Boosting</b>", td_style), Paragraph("<b>Cost-Sensitive (w=3.76)</b>", td_style), Paragraph("67.4%", td_style), Paragraph("64.9%", td_style), Paragraph("68.1%", td_style), Paragraph("<b>0.730</b>", td_style)],
        [Paragraph("Subset D (TF-IDF)", td_left), Paragraph("Gradient Boosting", td_style), Paragraph("TF-IDF 2000 (Cost-Sens)", td_style), Paragraph("68.1%", td_style), Paragraph("65.1%", td_style), Paragraph("68.9%", td_style), Paragraph("0.736", td_style)],
        [Paragraph("<b>Subset Context (23f)</b>", td_left), Paragraph("<b>Gradient Boosting (Context)</b>", td_style), Paragraph("<b>Cost-Sensitive (w=3.76)</b>", td_style), Paragraph("<b>74.6%</b>", td_style), Paragraph("<b>78.7%</b>", td_style), Paragraph("<b>73.6%</b>", td_style), Paragraph("<b>0.842</b>", td_style)],
        [Paragraph("<b>Subset Context (23f)</b>", td_left), Paragraph("<b>Stacking Meta-Learner</b>", td_style), Paragraph("<b>Ensemble (LR+GB+RF)</b>", td_style), Paragraph("<b>82.5%</b>", td_style), Paragraph("40.3%", td_style), Paragraph("<b>93.7%</b>", td_style), Paragraph("<b>0.841</b>", td_style)],
        [Paragraph("<b>Subset Context (23f)</b>", td_left), Paragraph("<b>PyTorch ReviewMLP</b>", td_style), Paragraph("<b>Cost-Weighted (w=3.76)</b>", td_style), Paragraph("<b>74.2%</b>", td_style), Paragraph("<b>78.8%</b>", td_style), Paragraph("73.0%", td_style), Paragraph("<b>0.841</b>", td_style)],
    ]

    t = Table(table_data, colWidths=[1.1*inch, 1.4*inch, 1.4*inch, 0.65*inch, 0.7*inch, 0.7*inch, 0.65*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Core Takeaways from Experimental Measurements:</b><br/>"
        "1. <b>The Accuracy Paradox in Action:</b> Baseline unweighted models hit ~80% accuracy, but Sensitivity is catastrophic (6.5% to 16.1%)! The model simply guesses 'Unpopular' almost every time. In an information retrieval application, missing 90% of viral reviews renders the system useless.<br/>"
        "2. <b>Cost-Sensitive Equivalence Without Data Loss:</b> Cost-sensitive loss weighting matches or exceeds random undersampling performance (0.730 vs 0.729 AUC; higher specificity 68.1% vs 67.1%) while keeping <b>100% of the training dataset</b> and avoiding the deletion of ~36,000 majority reviews.<br/>"
        "3. <b>The Contextual Breakthrough:</b> Shifting from Subset B (12 features) to Subset Context (23 features) drives ROC-AUC from <b>0.730 straight to 0.842</b>, while Sensitivity surges to <b>78.7%</b> (catching ~8 out of 10 viral reviews).<br/>"
        "4. <b>Stacking Ensemble Superiority (Unit 3):</b> Blending LR, HistGB, and Random Forest delivers the highest overall accuracy (**82.5%**) and exceptional specificity (**93.7%**), filtering out false alarms.",
        body
    ))
    story.append(Spacer(1, 4))

    # ================= SECTION 6: STUDENT ABLATIONS =================
    story.append(Paragraph("6. Curricular Student Experiments (Syllabus Units 1 & 2)", sec_title))
    story.append(Paragraph(
        "• <b>Experiment 1 (Unit 2 — ANN Architecture Depth & Learning Rate Sweep):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Tested 1-Layer (16 units), 2-Layer (32→16 + Dropout 0.2), and 3-Layer (64→32→16) on Contextual features under η ∈ {0.01, 0.001}.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>Finding:</i> 1-Layer was under-parameterized (Val AUC: 0.812). Both 2-Layer and 3-Layer networks converged stably to <b>0.835 – 0.836 Val AUC</b>. The 2-Layer ReviewMLP with 20% Dropout was proven optimal: it delivers maximum generalization with half the parameters and training time of the 3-Layer net.<br/>"
        "• <b>Experiment 2 (Unit 1 — Empirical Decision Tree Bias-Variance Tradeoff Curve):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;Swept Decision Tree depth from d = 1 to 15 on Contextual features.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<i>Finding:</i> Demonstrates textbook machine learning theory. At d=1 (stump), the tree suffers from <b>high bias / underfitting</b> (Train AUC 0.710, Test AUC 0.710). As capacity increases, it reaches the <b>optimal generalization sweet spot at d = 7 (Test AUC: 0.824)</b>. Beyond d ≥ 12, the model experiences <b>severe variance / overfitting</b>: Train AUC surges to 0.948 while Test AUC collapses to 0.694.",
        body
    ))
    story.append(Spacer(1, 4))

    # ================= SECTION 7: VIVA DEFENSE SCRIPT =================
    story.append(Paragraph("7. Anticipated Viva Q&A — Top 8 Questions & Exact Answers", sec_title))

    viva_qa = [
        ("Q1: What is 'like_share' and why not use raw like counts?",
         "Raw like counts are unfair across books. 50 likes on Harry Potter (which has 100,000 likes) is only 0.05% of engagement (Unpopular). But 50 likes on a niche indie book with 100 total likes is 50% of the book's total discussion (Highly Viral). like_share normalizes engagement by dividing review likes by the total likes across that entire book."),
        
        ("Q2: Why did the Stanford paper hit an AUC ceiling of 0.700, and how did you reach 0.842?",
         "The Stanford paper analyzed review text in a vacuum without book-level scale. We engineered 'user_to_book_ratio' [ln(user_reviews)/ln(ratings_count)] and 'contrarian_score' [rating_diff × log_words]. This boosted feature correlation with popularity from +0.22 to +0.47, driving ROC-AUC to 0.842."),
         
        ("Q3: Why use Cost-Sensitive Weighting instead of Random Undersampling?",
         "Random undersampling discards ~70% of majority class training reviews to create a 50:50 ratio, destroying valuable variance and edge cases. Cost-sensitive weighting keeps 100% of data and sets a penalty weight w = N_neg / N_pos ≈ 3.76 on positive errors. It achieves identical or better AUC without throwing away 36,000 reviews."),
         
        ("Q4: Why did you take log(1 + x) transforms on ratings_count and user_reviews?",
         "Real-world social engagement follows power laws. Book ratings range from 5 to 4,000,000+ (an 800,000x gap). Unscaled data crushes 99.2% of samples into the first bin and causes gradient explosions. Log transforms convert exponential data into smooth bell curves and model diminishing returns."),
         
        ("Q5: Why did you choose maximum tree depth of 5 to 7 in Gradient Boosting?",
         "In Experiment 2, we swept tree depth from 1 to 15. The empirical bias-variance curve peaked at depth 7 (Test AUC: 0.824) and overfit heavily at depth 12+ (Train AUC 0.948 vs Test AUC 0.694). Setting max_depth=5–6 in boosting ensures base estimators remain shallow weak learners that combine smoothly without overfitting."),
         
        ("Q6: How is your Stacking Classifier built, and why does it achieve 82.5% accuracy?",
         "Fulfilling Unit 3, it combines three distinct paradigms: Balanced Logistic Regression (linear boundary), HistGradientBoosting (non-linear thresholds), and Random Forest (bagging variance reduction). A Logistic Regression meta-classifier learns the optimal probability blending using 3-fold cross-validation out-of-fold predictions."),
         
        ("Q7: What does Reviewer Past Reviews = 30 mean in the interactive demo?",
         "It represents the user's lifetime total reviews written across their entire Goodreads profile (not reviews on that single book, since Goodreads only permits 1 review per user per book). It indicates whether the reviewer is a casual reader (10–30 reviews) or a platform influencer (1,000+ reviews)."),
         
        ("Q8: How does the interactive demo run in under 0.1 seconds?",
         "We serialized the trained Gradient Boosting model, StandardScaler, and the 23-feature schema into 'demo_model.pkl'. The script detects this binary artifact and loads it instantly without retraining, extracting NLTK POS tags and VADER scores on the fly from user text in real time.")
    ]

    for q, a in viva_qa:
        story.append(Paragraph(f"<b>{q}</b>", subsec_title))
        story.append(Paragraph(f"<i>Suggested Viva Answer:</i> \"{a}\"", body_tight))

    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#94A3B8'), spaceBefore=2, spaceAfter=4))
    story.append(Paragraph(
        "<i>Repository: https://github.com/Preksha-a11y/goodreads_ML | Master Runner: run_experiments.py | Interactive Demo: interactive_demo.py</i>",
        doc_subtitle
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated Guide PDF successfully: {pdf_path}")

if __name__ == '__main__':
    create_guide_pdf()
