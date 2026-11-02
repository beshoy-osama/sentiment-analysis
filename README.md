# IMDB Sentiment Analysis

> Binary sentiment classification on 50,000 IMDB movie reviews, comparing classical ML, recurrent deep learning, and a fine-tuned transformer.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing Pipelines](#preprocessing-pipelines)
- [Models](#models)
- [Results](#results)
- [Known Limitations](#known-limitations)
- [Installation](#installation)
- [Usage](#usage)
- [Saved Models](#saved-models)
- [Tech Stack](#tech-stack)

---

## Overview

This project tackles the classic NLP problem of **binary sentiment classification**: deciding whether a movie review is *positive* or *negative*. It implements and compares **six models across three paradigms**:

| Paradigm | Models |
|---|---|
| Classical ML | Logistic Regression, LinearSVC, Multinomial Naive Bayes |
| Deep Learning (Sequence) | Simple RNN, LSTM |
| Transformer | DistilBERT (fine-tuned) |

Each paradigm uses a dedicated preprocessing pipeline tuned to its needs, from aggressive TF-IDF-friendly cleaning to the minimal cleaning preferred by WordPiece tokenizers.

---

## Dataset

**[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)** (Kaggle)

| Attribute | Value |
|---|---|
| Total reviews | 50,000 |
| Duplicates removed | 418 exact duplicates |
| Reviews after deduplication | 49,582 (24,884 positive, 24,698 negative) |
| Class balance | Nearly balanced (50.2% positive / 49.8% negative) |
| Train / Test split | 80% / 20%, stratified (39,665 train, 9,917 test) |
| Test set composition | 4,940 negative, 4,977 positive |

The dataset is downloaded automatically at runtime via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
```

---

## Exploratory Data Analysis

Covered in **`src/notebooks/eda.ipynb`**. Key findings:

- **Class distribution**: nearly balanced after deduplication (50.2% / 49.8%); no resampling needed.
- **Review length**: mean 227 words, median 171, 95th percentile 581, 99th percentile 893, maximum 2,450. The distribution is heavily right-skewed.
- **Sentiment vs. length**: no consistent trend. The share of positive reviews stays between about 49% and 55% for reviews up to 1,000 words. Positive reviews are slightly longer only in the upper tail (95th percentile 597 vs. 562 words). The 1,000+ word bucket is 69% positive but contains only 42 reviews.
- **Vocabulary** (stop words removed, words of 3+ letters): 74,468 unique words in positive reviews and 71,829 in negative reviews; 45,558 are shared, 28,910 are exclusive to positive and 26,271 exclusive to negative.
- **Top words and n-grams**: positive reviews center around words like *film*, *great*, *love*, *performance*; negative reviews around *bad*, *worst*, *waste*, *plot*. Bigrams and trigrams reveal phrase-level patterns (e.g. *special effects*, *low budget*, *highly recommend*).

Visualizations use a dark-themed Matplotlib/Seaborn style with colour-coded sentiment (green = positive, red = negative).

---

## Preprocessing Pipelines

Three distinct pipelines are implemented, each optimized for its model family. All three share a universal cleaning layer (HTML tags, square-bracket content, and URLs removed).

| Step | ML Pipeline | DL Sequence Pipeline | Transformer Pipeline |
|---|---|---|---|
| Strip HTML tags | Yes | Yes | Yes |
| Remove square brackets | Yes | Yes | Yes |
| Remove URLs | Yes | Yes | Yes |
| Lowercase | Yes | Yes | No |
| Expand contractions | Yes (35-entry dictionary) | Yes | No |
| Remove special characters | Yes | Yes | No |
| Remove stop words | Yes (21 negation words kept) | No | No |
| POS-aware lemmatization | Yes | Yes | No |

**Key design decisions:**

- **Negation preservation**: negation words such as *not*, *never*, and *cannot* are excluded from the stop-word list in the ML pipeline. Removing them would flip sentiment meaning (e.g. "not good" becoming "good").
- **DL keeps context words**: RNN/LSTM rely on sequential context, so stop words are kept for the embedding layer.
- **Transformers get minimal cleaning**: DistilBERT's WordPiece tokenizer was pre-trained on raw text and handles casing, punctuation, and sub-words natively.

---

## Models

### Classical ML: TF-IDF + Classifiers

TF-IDF vectorization with unigrams + bigrams at two vocabulary sizes (10k and 50k features). The 50k configuration scored higher for all three models and is used in the results below.

- **Logistic Regression**: strong, interpretable baseline
- **LinearSVC**: best-scoring model overall; saved to disk
- **Multinomial Naive Bayes**: simplest model; requires non-negative TF-IDF inputs

### Deep Learning: Sequence Models

```
Embedding (vocab=50k, dim=64) -> [RNN | LSTM](64 units) -> Dense(16/32, ReLU) -> Dropout -> Dense(1, Sigmoid)
```

- Max sequence length: **580 tokens** (about the 95th percentile of review length: 581 words in the EDA)
- Optimizer: Adam; Loss: Binary Crossentropy
- Early stopping on `val_loss` with `patience=3`, restoring best weights. Early stopping ended Simple RNN training at epoch 10 (best epoch 7) and LSTM training at epoch 7 (best epoch 4).
- The validation data is the held-out test set (see [Known Limitations](#known-limitations)).

### Transformer: DistilBERT Fine-Tuning

Fine-tunes `distilbert-base-uncased` (66M parameters) for sequence classification with the Hugging Face `Trainer`:

```
[CLS] token -> DistilBERT encoder (6 layers, 768-dim) -> Classification head (2 labels)
```

| Hyperparameter | Value |
|---|---|
| Max token length | 256 (about 190 words; longer reviews are truncated) |
| Epochs | Up to 8, with early stopping on `eval_loss` (patience=2); training stopped after 3 epochs (1,860 steps, about 14.5 minutes) |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.05 |
| Warmup | `warmup_steps=0.1` in the notebook (intended as a 10% ratio; see Known Limitations) |
| Mixed precision (fp16) | Enabled if a GPU is available |

---

## Results

Evaluated on a held-out test set of **9,917 reviews** (4,940 negative, 4,977 positive) from an 80/20 stratified split. Precision, recall, and F1 are weighted averages over the two classes.

| Model | Features | Accuracy | Exact accuracy | Precision | Recall | F1 | Neg. recall | Pos. recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | TF-IDF 50k | 90% | 90.1% | 0.90 | 0.90 | 0.90 | 89.1% | 91.1% |
| **Linear SVC** | TF-IDF 50k | **91%** | **90.5%** | 0.91 | 0.91 | **0.91** | 89.8% | 91.2% |
| Naive Bayes | TF-IDF 50k | 88% | 88.3% | 0.88 | 0.88 | 0.88 | 87.6% | 89.0% |
| Simple RNN | none | 83% | 82.6% | 0.83 | 0.83 | 0.83 | 86.4% | 78.9% |
| LSTM | none | 90% | 89.9% | 0.90 | 0.90 | 0.90 | 89.1% | 90.7% |
| DistilBERT | none | 90% | 90.3% | 0.91 | 0.90 | 0.90 | 87.1% | **93.6%** |

> The 10k TF-IDF configuration scored lower than 50k for all three classical models.

### Key Observations

- **Linear SVC has the highest accuracy (90.5%) and weighted F1 (0.91)**, and as a TF-IDF plus linear model it is lightweight, making it a strong candidate for latency-sensitive use. Training and inference times were not benchmarked.
- **DistilBERT has the highest positive recall (93.6%)** but a lower negative recall (87.1%). Its accuracy (90.3%) is effectively tied with Linear SVC: the difference is 19 reviews out of 9,917.
- **Differences of about one point among the top four models are within sampling noise.** With 9,917 test reviews, the 95% confidence interval on accuracy is roughly ±0.6 percentage points, and the numbers come from a single run.
- **LSTM matches Logistic Regression (about 90%)** and outperforms Simple RNN by about 7 points, consistent with the benefit of gating for long reviews.
- **Simple RNN (83%) should not be deployed**: it falls below every other model.

### Confusion Matrix Insight

Logistic Regression, Linear SVC, Naive Bayes, LSTM, and DistilBERT all make more false positives (negative reviews predicted positive) than false negatives, a common pattern on IMDB due to mixed-sentiment language in critical reviews. DistilBERT shows the strongest skew (639 false positives vs. 319 false negatives), which is the other side of its high positive recall. Simple RNN is the exception, with more false negatives (1,051) than false positives (673).

---

## Known Limitations

- **Test set used during model selection.** Early stopping for the RNN/LSTM and `load_best_model_at_end` plus early stopping for DistilBERT monitored the same held-out test set that is reported above, and the 10k vs. 50k TF-IDF comparison was also made on it. Reported metrics may therefore be slightly optimistic. A separate train/validation/test split is recommended.
- **Single run, no confidence intervals or multiple seeds.**
- **DistilBERT truncation and warmup.** Inputs are truncated at 256 tokens, which cuts more than a quarter of reviews (the 75th percentile is 276 words). The notebook sets `warmup_steps=0.1`, but a 10% warmup would need `warmup_ratio=0.1`; depending on the `transformers` version, 0.1 steps may effectively disable warmup. Both may have limited its accuracy.
- **No systematic hyperparameter search**; only the TF-IDF vocabulary size (10k vs. 50k) was compared.
- **Results are specific to the IMDB domain** and may need domain adaptation for other review types.
- **No latency or training-time benchmarks** were run for the classical models.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU *(recommended for transformer fine-tuning)*

> `src/app/app.py` includes a compatibility loader for `.keras` models saved with Keras 3 when running on TensorFlow 2.10.

### 1. Clone the repository

```bash
git clone https://github.com/beshoy-osama/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the Gradio app and the core libraries:

```
gradio>=4.0
numpy
pandas
scikit-learn
joblib
beautifulsoup4
nltk
tensorflow
torch
transformers
safetensors
```

To run the notebooks you also need:

```bash
pip install matplotlib seaborn wordcloud datasets kagglehub
```

---

## Usage

### Run EDA

```bash
jupyter notebook src/notebooks/eda.ipynb
```

### Run the full modeling pipeline

```bash
jupyter notebook src/notebooks/sentiment_analysis_imdb.ipynb
```

Run the notebook through the save cells to generate `src/saved_models/`.

### Run the demo app

```bash
python src/app/app.py
```

Then open http://127.0.0.1:7860.

---

## Saved Models

The notebook saves trained models to `src/saved_models/` (excluded from git). The demo app expects this layout:

```
src/saved_models/
├── ml/
│   ├── linear_svc_model.joblib
│   └── tfidf_50k_vectorizer.joblib
├── dl_lstm/
│   ├── lstm_sentiment_model.keras
│   ├── tokenizer_dl.joblib
│   └── dl_config.json
└── transformer_distilbert/           # Hugging Face model + tokenizer files
```

The app needs these files to produce predictions. Generate them by running the notebook, or place pre-trained copies in this layout.

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy`, `kagglehub` |
| Text Processing | `nltk`, `beautifulsoup4`, `scikit-learn` |
| Classical ML | `scikit-learn` (LogisticRegression, LinearSVC, MultinomialNB) |
| Deep Learning | `tensorflow` / `keras` (Embedding, SimpleRNN, LSTM) |
| Transformers | `transformers`, `datasets`, `torch` |
| Serving | `gradio`, `joblib` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |

---

*Dataset credit: [Lakshmi Pathi, Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)*
