---
title: "A Configurable Traditional Text Classification Pipeline for Fine-Grained Banking Intent Detection"
subtitle: "Machine Learning Course Assignment — Task 2 (Text Data) with Bonus Word2Vec + BiLSTM"
author:
  - "[Member 1 full name] ([Student ID]) · [Member 2 full name] ([Student ID])"
date: "Semester I, Academic Year 2026–2027"
geometry: margin=2cm
fontsize: 10pt
colorlinks: true
toc: true
toc-depth: 2
numbersections: true
header-includes:
  - \usepackage{etoolbox}
  - \AtBeginEnvironment{longtable}{\small}
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

\newpage

# Course and group information

| | |
| --- | --- |
| University / Faculty | Ho Chi Minh City University of Technology (HCMUT), VNU-HCM — Faculty of Computer Science and Engineering |
| Course | Machine Learning, [Course code, e.g. CO3117] |
| Semester | Semester I, Academic Year 2026–2027 |
| Instructor | [Instructor full name and title] ([instructor email]) |
| Topic | Task 2 — Machine Learning with Text Data, including a bonus Word2Vec + BiLSTM pipeline |
| Dataset | Banking77 (`mteb/banking77`) |
| Colab notebook | `notebooks/text_classification_colab.ipynb` |
| Source repository | [GitHub repository URL] |
| Extracted features | `features/banking77/` (`.npy`, `.npz`, `.h5`) |

| Full name | Student ID | Email |
| --- | --- | --- |
| Dương Minh Nhật | 2452889 | [nhat.duongminhnickct@hcmut.edu.vn] |
| Nguyễn Trọng Nhân | 2452883 | [member2@hcmut.edu.vn] |

Individual contribution percentages are reported in Section~10 (Work distribution).

# Abstract

This report presents a complete traditional machine learning pipeline for text classification on the Banking77 intent-detection benchmark, together with a bonus deep-learning comparison, as required by Task~2 of the Machine Learning course assignment. Configurable preprocessing (stop-word removal, tokenization, padding, and optional negation marking) is combined with three feature families—TF-IDF (uni- and bigrams), mean-pooled Word2Vec, and frozen DistilBERT—whose representations are persisted as `.npy` / `.h5` files and classified with Naive Bayes, logistic regression, and a calibrated linear SVM. A bonus bidirectional LSTM over the same Word2Vec table evaluates whether sequence modelling recovers information discarded by bag averaging.

On the official Banking77 test split, sparse TF-IDF with a linear SVM attains the highest Macro-F1 (89.18), outperforming frozen DistilBERT + logistic regression (87.29). Mean-pooled Word2Vec is substantially weaker under linear heads (best 78.43 Macro-F1), while Word2Vec + BiLSTM recovers to 83.07 Macro-F1, illustrating the value of order for short queries. Preprocessing ablations show that stop-word removal harms both TF-IDF and DistilBERT, that subword tokenization and masked padding are critical for DistilBERT (+14.4 and +18.3 Macro-F1 points), and that negation marking improves BiLSTM by +4.1 Macro-F1. Calibration metrics further separate discrimination from probability quality. The study links these outcomes to course concepts: conditional independence, margin-based linear classification, static versus contextual embeddings, and sequence models.

# Introduction

## Motivation and assignment alignment

The Machine Learning course assignment requires students to apply theoretical knowledge to a full pipeline: exploratory data analysis (EDA), preprocessing, feature extraction, model training, and evaluation. Task~2 focuses on text data and mandates (i) a traditional pipeline with configurable preprocessing and classifiers, (ii) at least two feature extractors spanning lexical and modern embedding approaches, with embeddings saved as `.npy` or `.h5`, and (iii) an EDA report plus a comparison of traditional and modern methods. An optional deep-learning pipeline earns bonus credit when compared against the traditional path.

We concentrate on a single, demanding corpus—**Banking77**—so that every design choice can be motivated by EDA and interpreted in depth. Banking77 comprises 77 fine-grained online-banking intents. Many classes share content nouns (*card*, *transfer*, *top-up*) and differ by function words, polarity, or transactional state (*pending*, *failed*, *not received*). This regime stresses both sparse lexical models and frozen contextual encoders used without task-specific fine-tuning.

## Research questions

1. **When do lexical features suffice for fine-grained intent?** Does TF-IDF with a linear classifier remain competitive with frozen Transformer embeddings on a 77-way problem?
2. **What does frozen DistilBERT contribute without fine-tuning?** Does masked-language pretraining alone yield linearly separable intent geometry under logistic regression or a linear SVM?
3. **How should preprocessing interact with representation?** Do stop-word removal, subword tokenization, padding/masking, and negation marking affect TF-IDF, frozen DistilBERT, and Word2Vec sequence models differently?
4. **What does a bonus BiLSTM add over mean-pooled Word2Vec?** Relative to bag averaging of static embeddings, how much does bidirectional sequence modelling recover on short banking queries?

## Contributions

- A configurable traditional pipeline covering EDA, Boolean preprocessing, three embedding families, three classical classifier families, and calibrated evaluation (accuracy, Macro-F1, Brier score, ECE).
- Persistent feature artefacts in `.npy` / `.h5` that feed identical classifiers across configurations.
- A focused Banking77 study with one-flag preprocessing ablation and error / negation analysis.
- A bonus Word2Vec + BiLSTM path that isolates the effect of order under a shared static embedding table.

# Theoretical background

## Lexical representations

**TF-IDF** reweights term frequencies by inverse document frequency so that rare, discriminative words receive higher weight. **$n$-grams** (here, unigrams and bigrams) partially recover local order (*not arrived*, *top up*). These features are sparse, interpretable, and effective when class labels correlate with specific lexical patterns—even when those patterns are short phrases rather than single topic keywords.

## Classical classifiers

**Multinomial Naive Bayes** models $P(x \mid y)$ under a conditional independence assumption over features. On sparse word counts this assumption is often useful; on dense, highly correlated embedding dimensions, Gaussian Naive Bayes tends to over-count evidence and produce overconfident posteriors.

**Logistic regression** and the **linear SVM** are discriminative models. Logistic regression estimates $P(y \mid x)$ via a linear (one-vs-rest) decision boundary; the linear SVM maximises margin. Both assume that classes are approximately linearly separable in feature space. Probabilities for the SVM are obtained by Platt scaling.

## Static and contextual embeddings

**Word2Vec** (skip-gram) learns fixed word vectors from local co-occurrence. Averaging word vectors yields a document embedding that ignores order; feeding the sequence into a BiLSTM restores order while still relying on corpus-trained static geometry.

**DistilBERT** is pretrained with a masked language modelling (MLM) objective: predict masked tokens from context. Mean-pooling its token states produces a document vector, but MLM does not explicitly optimise sentence-level similarity or intent separation. Using DistilBERT as a **frozen** feature extractor therefore tests whether MLM geometry is already linearly useful for Banking77, without the confounding effect of end-to-end fine-tuning.

## Sequence models

A **bidirectional LSTM** summarises left-to-right and right-to-left context. In the bonus pipeline it consumes Word2Vec embeddings of the token sequence, with packing enabled when the `padding` preprocess switch is on. This design attributes gains over mean pooling to sequence modelling rather than to a different word representation.

## Evaluation metrics

**Accuracy** treats all examples equally. **Macro-averaged F1** averages per-class F1 and is appropriate for Banking77, where intents are imbalanced (35–187 training examples per class). **Brier score** and **expected calibration error (ECE)** assess whether predicted probabilities match empirical frequencies.

# Datasets and exploratory data analysis

## Dataset

Banking77 is downloaded from the public Hugging Face identifier `mteb/banking77` inside the notebook; no personal cloud drive is mounted. We use the official train / test split (9,993 / 3,076). Test data are never used for vocabulary fitting, PCA, early stopping, or model selection.

| Dataset | Hugging Face id | Task | Classes | Train / test |
| ---------- | ----------------- | -------------------------------- | ------: | ------------ |
| Banking77 | `mteb/banking77` | online-banking customer intent | 77 | 9,993 / 3,076 |

## Descriptive statistics

Word counts use normalised text (lower case; HTML and URLs removed) and whitespace tokenization.

| Statistic | Banking77 |
| ---------------------------------- | -----------------: |
| training / test documents | 9,993 / 3,076 |
| documents per class (min / max) | 35 / 187 |
| balance ratio (min/max class count) | 0.187 |
| words per document: mean (std) | 11.96 (7.89) |
| median / 95th pct / 99th pct / max | 10 / 29.4 / 43 / 79 |
| vocabulary with / without stop words | 2,353 / 2,128 |
| share of tokens that are stop words | 60.7% |
| documents containing a negation | 25.5% |

![Class balance of the Banking77 training set (ordered by frequency).](figures/banking77/class_balance.png){width=70%}

![Document length distribution with median and 95th-percentile cut-offs used for padding.](figures/banking77/length_hist.png){width=70%}

![Top-15 words with and without stop words. Function words and question cues are frequent.](figures/banking77/top_words_comparison.png){width=85%}

## Hypotheses derived from EDA

- **H1 (lexical overlap).** Intents share content nouns; unigram topic cues alone are insufficient, but TF-IDF bigrams and linear margins may still separate many classes.
- **H2 (function words matter).** Stop words comprise about 61% of tokens and often encode intent (*from*, *into*, *not*, *why*, *how*). Removing them should degrade TF-IDF and DistilBERT.
- **H3 (negation and state).** Negation appears in 25.5% of training documents and frequently marks failure or pending states. Explicit negation marking should help static sequence models more than frozen contextual encoders.
- **H4 (length and padding).** Short queries (median 10 words) make truncation-to-10-tokens and unmasked padding particularly harmful for neural encoders; masked padding to the 95th percentile should improve Macro-F1 substantially for DistilBERT and BiLSTM.

# Methodology

## Overview of the traditional and bonus pipelines

The implementation separates a **traditional path** (configurable preprocessing -> feature extraction to disk -> classical classifiers -> evaluation) from a **bonus deep-learning path** (Word2Vec + BiLSTM on the same preprocessed text). Configurations are encoded by dataclasses (`PreprocessConfig`, `ExperimentConfig`). Results are cached by dataset, preprocessing tag, and method. Random seeds are fixed to 42.

![Overview of the traditional and bonus text-classification pipelines on Banking77.](figures/pipeline_overview.png){width=90%}

## Preprocessing

All text is lower-cased and stripped of HTML, URLs, and related artefacts. Four Boolean switches control the remainder:

| Switch | Off | On |
| ---------------- | ---------------------------------------- | -------------------------------------------- |
| `stop_word_removal` | retain all tokens | remove scikit-learn `ENGLISH_STOP_WORDS` |
| `tokenization` | whitespace unigrams / whole-word lookup | regex tokens + bigrams for TF-IDF; WordPiece for DistilBERT |
| `padding` | truncate to 10 tokens; no masking | pad/truncate to the 95th-percentile length with attention / packing masks |
| `negation_marking` | unchanged text | join a negation cue with the next content token (`not arrived` -> `not_arrived`) |

The default for the main comparison is `sw0_tok1_pad1_neg0`. Padding is not applicable to bag-of-words / TF-IDF models.

## Feature extraction and persistence

| Key | Family | Representation |
| -------- | ---------- | ---------------------------------------------------- |
| `tfidf` | traditional | `TfidfVectorizer` (sublinear TF, max 25k features, min_df=2), uni+bigrams |
| `word2vec` | static | skip-gram Word2Vec on the training split (300-d); mean-pooled documents |
| `distilbert` | modern | frozen DistilBERT; masked mean pool (768-d) |

Vectorizers, Word2Vec, and any PCA/SVD projections are fit on training data only. Each feature directory stores dense `.npy` or sparse `.npz` arrays, labels, a consolidated `features.h5`, and `meta.json`, satisfying the assignment requirement to persist embeddings for reuse by classifiers. BoW remains available as a plug-in embedding type in the module API but is not part of the default experiment grid.

## Classical classifiers

| Classifier | Sparse input (TF-IDF) | Dense input (embeddings) |
| ---------------- | ---------------------------------- | ------------------------ |
| Naive Bayes | MultinomialNB ($\alpha=0.5$) | GaussianNB |
| Logistic regression | $C=2.0$, balanced, L-BFGS | StandardScaler + same |
| Linear SVM | LinearSVC ($C=1.0$, balanced) | L2 Normalizer + same |

Dense SVM inputs are L2-normalised for stable convergence. SVM probabilities use Platt scaling (3-fold).

## Bonus deep-learning model

The bonus component is **Word2Vec + BiLSTM**: the same skip-gram table used for mean pooling initialises a trainable embedding layer; a one-layer bidirectional LSTM (hidden size 192) produces the document representation. Early stopping monitors a stratified 10% validation split of the training set (patience 2, up to 7 epochs).

## Evaluation protocol

Models train on the official training split and are evaluated once on the held-out test split. We report accuracy, macro-averaged precision, recall, and F1, and calibration scores

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\left(p_{ik}-y_{ik}\right)^2, \qquad
\text{ECE} = \sum_{b=1}^{15}\frac{|B_b|}{N}\left|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\right|.$$

# Experimental results

## Main comparison

**Expectation.** Under H1, TF-IDF should be strong but not necessarily dominant. Under H2–H4, DistilBERT should depend heavily on tokenization and padding; BiLSTM should outperform mean-pooled Word2Vec.

All rows use `sw0_tok1_pad1_neg0`. Bold marks the best Macro-F1.

| Method | Acc | Macro-F1 | Brier | ECE | Train (s) |
| ------------------------------ | -----: | -------: | -----: | -----: | ------: |
| TF-IDF + MultinomialNB | 82.35 | 81.54 | 0.440 | 0.386 | 0.01 |
| TF-IDF + LogReg | 88.62 | 88.65 | 0.315 | 0.329 | 2.10 |
| TF-IDF + LinearSVC | 89.17 | **89.18** | 0.181 | 0.127 | 1.30 |
| Word2Vec + GaussianNB | 63.17 | 63.10 | 0.704 | 0.346 | 0.01 |
| Word2Vec + LogReg | 78.41 | 78.43 | 0.331 | 0.099 | 1.47 |
| Word2Vec + LinearSVC | 74.61 | 74.43 | 0.440 | 0.250 | 20.12 |
| DistilBERT + GaussianNB | 70.61 | 70.69 | 0.574 | 0.284 | 0.02 |
| DistilBERT + LogReg | 87.32 | 87.29 | 0.205 | 0.072 | 0.67 |
| DistilBERT + LinearSVC | 85.99 | 85.97 | 0.240 | 0.157 | 50.28 |
| Word2Vec + BiLSTM (bonus) | 83.13 | 83.07 | 0.242 | 0.019 | 62.49 |

![Accuracy and Macro-F1 on Banking77.](figures/banking77/accuracy_f1_bars.png){width=80%}

![Macro-F1 heatmap of the default configuration.](figures/master_f1_heatmap.png){width=55%}

## Traditional versus modern approaches

| Best traditional (sparse / static) | Best modern (frozen Transformer) | Gap (pp) | Bonus DL |
| ---------------------------------- | -------------------------------- | -------: | ------------------------ |
| TF-IDF + LinearSVC, 89.18 | DistilBERT + LogReg, 87.29 | **−1.89** (traditional ahead) | Word2Vec + BiLSTM, 83.07 |

**Interpretation.** On Banking77, a well-tuned sparse pipeline remains superior to a *frozen* DistilBERT encoder under linear heads. This does not claim that Transformers are inherently unsuitable for intent detection; rather, it shows that MLM pretraining alone, without sentence-level contrastive training or task fine-tuning, does not guarantee better linear separability than TF-IDF bigrams on short, lexically overlapping queries. Mean-pooled Word2Vec is clearly inadequate (78.43 Macro-F1 at best). The BiLSTM recovers about 4.6 Macro-F1 points over Word2Vec + LogReg, confirming H4’s emphasis on order, yet still trails TF-IDF + LinearSVC by about 6.1 points.

## Preprocessing ablation

Each cell reports $\Delta$Macro-F1 $= \mathrm{F1}(\mathrm{flag}=\mathrm{True}) - \mathrm{F1}(\mathrm{flag}=\mathrm{False})$ in percentage points. Ablation models: TF-IDF + LogReg, DistilBERT + LogReg, Word2Vec + BiLSTM.

| Model | `stop_word_removal` | `tokenization` | `padding` | `negation_marking` |
| -------------------------- | -----------------: | -------------: | ---------: | -----------------: |
| TF-IDF + LogReg | −4.05 | +1.42 | n/a | −0.04 |
| DistilBERT + LogReg | −3.54 | **+14.37** | **+18.26** | +0.23 |
| Word2Vec + BiLSTM | +0.72 | −0.80 | **+5.33** | **+4.13** |

![Preprocessing ablation heat map (blue = enabling the flag improves Macro-F1).](figures/banking77/preprocessing_ablation_heatmap.png){width=85%}

**Findings.** Stop-word removal supports H2: TF-IDF and DistilBERT lose 3.5–4.0 Macro-F1 points when function words are deleted. Subword tokenization and masked padding are essential for DistilBERT (+14.4 and +18.3 points), supporting H4. Negation marking yields +4.13 Macro-F1 for Word2Vec + BiLSTM with near-zero effect on TF-IDF and DistilBERT—consistent with H3 and with the idea that contextual encoders already encode local negation to some degree, while static sequence models benefit from explicit scope marking.

## Calibration

![Macro-F1 versus Brier score on Banking77.](figures/banking77/brier_calibration_vs_f1.png){width=75%}

![Reliability diagrams for selected models.](figures/banking77/reliability_diagram.png){width=70%}

Multinomial Naive Bayes on TF-IDF reaches Macro-F1 81.54 with ECE 0.386—strong discrimination relative to Word2Vec, yet poorly calibrated. DistilBERT + LogReg achieves Macro-F1 87.29 with Brier 0.205 and ECE 0.072. BiLSTM is remarkably well calibrated (ECE 0.019) despite trailing TF-IDF in Macro-F1. These results reinforce that course evaluation must separate discrimination from probability quality.

## Naive Bayes on sparse versus dense features

Multinomial Naive Bayes remains a nearly free baseline on sparse TF-IDF (81.54 Macro-F1). Gaussian Naive Bayes collapses on Word2Vec (63.10) and is only moderately better on DistilBERT (70.69), matching the theoretical warning about correlated dense coordinates and the independence assumption.

## Bonus deep-learning pipeline

![BiLSTM training dynamics (validation loss versus epochs).](figures/banking77/bilstm_loss.png){width=80%}

| Representation + head | Macro-F1 |
| -------------------------------- | -------: |
| Word2Vec mean-pool + LogReg | 78.43 |
| Word2Vec mean-pool + LinearSVC | 74.43 |
| Word2Vec + BiLSTM | **83.07** |
| TF-IDF + LinearSVC (traditional best) | 89.18 |
| DistilBERT + LogReg (modern best) | 87.29 |

Under matched Word2Vec embeddings, the BiLSTM clearly outperforms bag averaging. Gains are largest when padding/masking and negation marking are enabled (ablation). At this labelled-data scale, however, a sparse TF-IDF linear SVM remains the overall optimum—an empirical reminder that “deep” is not automatically better when the lexical signal is strong and the encoder is frozen.

## Error and negation analysis

![Confusion structure for the best model (TF-IDF + LinearSVC).](figures/banking77/confusion_best.png){width=70%}

Frequent confusions cluster in near-synonymous intent families: identity verification pairs (`unable_to_verify_identity` / `verify_my_identity` / `why_verify_identity`), exchange-rate variants, and top-up / transfer state distinctions (`pending_top_up` vs `top_up_failed`, `pending_transfer` vs `failed_transfer`). These errors are semantically coherent: the models capture the topical domain but struggle with subtle transactional status.

| Method | Acc. with negation ($n=692$) | Acc. without ($n=2384$) | Gap |
| ---------------------------- | ---------------------------: | ----------------------: | ---------: |
| TF-IDF + MultinomialNB | 77.46 | 83.77 | −6.31 |
| TF-IDF + LinearSVC | 86.71 | 89.89 | −3.19 |
| DistilBERT + LogReg | 83.82 | 88.34 | −4.52 |
| Word2Vec + BiLSTM | 77.89 | 84.65 | −6.76 |

Negated queries remain harder for nearly all models. Combined with the ablation gains of `negation_marking` for BiLSTM (+4.13 Macro-F1), this supports H3: negation is a structural difficulty that lightweight scope marking can partially mitigate for static sequence models.

# Discussion

## Advantages and limitations by method family

| Approach | Advantages | Limitations |
| -------------------- | ---------------------------------------- | ---------------------------------------- |
| TF-IDF + linear model | Best Banking77 Macro-F1; fast; interpretable bigrams | Sensitive to stop-word removal; confuses near-synonymous intents |
| Multinomial NB | Strong sparse baseline; millisecond training | Poor calibration (ECE 0.39); weaker on overlapping intents |
| Word2Vec (mean-pool) | Inexpensive dense baseline; reusable for BiLSTM | Loses order; weak under linear heads on Banking77 |
| Logistic regression / LinearSVC | Robust discrimination; Platt-calibrated SVM probabilities | Fixed regularisation; SVM slower on high-dimensional dense × many classes |
| Frozen DistilBERT (MLM) | Strong when tokenized and padded correctly; good LogReg calibration | Trails TF-IDF without fine-tuning; collapses if padding/tokenization are wrong |
| Word2Vec + BiLSTM | Recovers order; excellent ECE; benefits from negation marking | Trails TF-IDF; longer training; single-seed variance |

## Learning outcomes relative to course concepts

The experiments confirm several textbook predictions and qualify others. The Naive Bayes independence assumption remains useful for sparse lexical features but fails as a probability model on dense embeddings. Margin-based linear classifiers excel when the feature space already encodes discriminative $n$-grams (TF-IDF). Transfer learning via a *frozen* MLM encoder is not automatically superior to classical features on short, fine-grained intents—task structure and fine-tuning matter. Sequence modelling over static embeddings improves upon bag averaging but does not close the gap to TF-IDF here. Finally, evaluation must include calibration: BiLSTM’s low ECE coexists with lower Macro-F1 than TF-IDF + LinearSVC.

# Conclusions

1. **On Banking77, traditional TF-IDF + LinearSVC is the strongest configuration studied** (89.18 Macro-F1), ahead of frozen DistilBERT + LogReg (87.29) and Word2Vec + BiLSTM (83.07).
2. **Preprocessing is representation-dependent.** Stop-word removal harms TF-IDF and DistilBERT. Subword tokenization and masked padding are critical for DistilBERT (+14.4 / +18.3 Macro-F1). Negation marking improves Word2Vec + BiLSTM by +4.13 Macro-F1.
3. **Bonus BiLSTM occupies a clear niche** over mean-pooled Word2Vec (+4.6 Macro-F1 vs LogReg) with strong calibration, but does not replace sparse linear models on this corpus.
4. **Errors concentrate in near-synonymous intent families**, and negated queries remain harder across methods.

**Limitations and future work.** Results use a single seed and fixed hyperparameters. DistilBERT is frozen; end-to-end fine-tuning or contrastive sentence encoders would likely change the traditional–modern ranking. The `padding=False` condition jointly disables length and masking.

# Work distribution

Individual grade $=$ group grade $\times$ contribution percentage. Both members participated in design meetings, code review, and report writing; the table lists primary ownership.

| Member | Primary responsibilities | Contribution |
| ------------------------ | ---------------------------------------------------------------- | -----------: |
| [Member 1 full name] ([Student ID]) | Dataset loading; Boolean preprocessing including negation marking; EDA; TF-IDF features; Naive Bayes, logistic regression, and calibrated SVM; preprocessing ablation; error and negation analysis; README / repository hygiene; report sections on EDA, classical methods, and ablation | 50\% |
| [Member 2 full name] ([Student ID]) | DistilBERT and Word2Vec feature extraction with `.npy` / `.h5` persistence; bonus Word2Vec + BiLSTM; evaluation suite and figures; Colab notebook; report sections on embeddings, deep learning, discussion, and conclusions | 50\% |

Collaboration evidence (meeting notes, screenshots) is archived under `reports/evidence/`.

# Reproducibility and deliverables

This submission follows Section~5 of the course assignment. The required directory layout is:

```
notebooks/     Colab front end (Runtime -> Run all)
modules/       supporting Python package imported by the notebook
reports/       PDF report, CSV tables, figures, collaboration evidence/
features/      extracted embeddings (.npy / .npz / .h5)
```

Additional details:

- **Colab front-end.** `notebooks/text_classification_colab.ipynb` runs without mounting personal drives; Banking77 is fetched from a public Hugging Face link; dependencies install inside the notebook.
- **Modules.** Code in `modules/` is imported by the notebook and by `main.py`.
- **PDF report.** The present document (EDA, pipeline design, experiments, comparisons, analysis, work distribution).
- **Feature artefacts.** `features/banking77/<embedding>_<tag>/` with `.npy`, `.npz`, `.h5`, and `meta.json`.
- **Tables and figures.** `comparison_banking77.csv`, `master_comparison.csv`, `master_f1_pivot.csv`, `master_ablation.csv`, `eda_summary.csv`, `errors_banking77.csv`, `negation_banking77.csv`; figures under `reports/figures/banking77/` and `pipeline_overview.png`.
- **Command line.** `python main.py` (full Banking77 suite); `python main.py --quick` for a smoke test.
- **Software.** Python 3.x, PyTorch, scikit-learn, transformers, datasets, NumPy Word2Vec training path.
- **GitHub (encouraged).** Course metadata, member details, execution guide, and links to the report and Colab should appear in the repository README.

# References

1. I. Casanueva, T. Temcinas, D. Gerz, M. Henderson, I. Vulic. *Efficient Intent Detection with Dual Sentence Encoders.* NLP4ConvAI 2020 (Banking77).
2. T. Mikolov, K. Chen, G. Corrado, J. Dean. *Efficient Estimation of Word Representations in Vector Space* (Word2Vec). 2013.
3. V. Sanh, L. Debut, J. Chaumond, T. Wolf. *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.* 2019.
4. G. Salton, C. Buckley. *Term-weighting approaches in automatic text retrieval.* Information Processing & Management, 1988 (TF-IDF).
5. C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017.
6. G. W. Brier. *Verification of forecasts expressed in terms of probability.* Monthly Weather Review, 1950.
7. J. Platt. *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods.* 1999.
8. S. Hochreiter, J. Schmidhuber. *Long Short-Term Memory.* Neural Computation, 1997.
