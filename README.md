# 🎬 IMDB Sentiment Analysis

> Binary sentiment classification on 50,000 movie reviews using classical ML, deep learning, and transformer models.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing Pipelines](#preprocessing-pipelines)
- [Models](#models)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Saved Models](#saved-models)
- [Tech Stack](#tech-stack)

---

## Overview

This project tackles the classic NLP problem of **binary sentiment classification** — determining whether a movie review is *positive* or *negative*. It implements and compares six model families across three paradigms:

| Paradigm | Models |
|---|---|
| Classical ML | Logistic Regression, LinearSVC, Multinomial Naive Bayes |
| Deep Learning (Sequence) | Simple RNN, LSTM |
| Transformer | DistilBERT (fine-tuned) |

Each paradigm uses a dedicated preprocessing pipeline tuned to its architectural needs, from aggressive TF-IDF-friendly cleaning to the minimal touch preferred by WordPiece tokenizers.

---

## Project Structure

```
.
├── eda.ipynb                          # Exploratory Data Analysis
├── sentiment_analysis_imdb_final.ipynb  # Full modeling pipeline
├── saved_models/
│   ├── ml/
│   │   ├── linear_svc_model.joblib
│   │   └── tfidf_50k_vectorizer.joblib
│   ├── dl_lstm/
│   │   ├── lstm_sentiment_model.keras
│   │   ├── tokenizer_dl.joblib
│   │   └── dl_config.json
│   └── transformer_distilbert/
│       ├── model files (HuggingFace format)
│       └── tokenizer files
└── README.md
```

---

## Dataset

**[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)** — Kaggle

| Attribute | Value |
|---|---|
| Total reviews | 50,000 |
| Classes | `positive` / `negative` |
| Class balance | 50% / 50% (perfectly balanced) |
| Duplicates removed | ~418 rows |
| Train / Test split | 80% / 20% (stratified) |

The dataset is downloaded automatically at runtime via `kagglehub`:

```python
import kagglehub
path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
```

---

## Exploratory Data Analysis

Covered in **`eda.ipynb`**. Key findings:

- **Class distribution** — perfectly balanced; no resampling needed.
- **Review length** — median ~230 words; 95th percentile ~580 words. Heavy right-skew with outliers beyond 2,000 words.
- **Sentiment vs length** — positive reviews trend slightly longer than negative ones.
- **Top words** — after stopword removal, positive reviews center around words like *film*, *great*, *love*, *performance*; negative reviews around *bad*, *worst*, *waste*, *plot*.
- **N-grams** — bigrams and trigrams reveal rich phrase-level patterns (e.g. *special effects*, *low budget*, *highly recommend*).

Visualizations use a dark-themed Matplotlib/Seaborn style with colour-coded sentiment (🟢 positive, 🔴 negative).

---

## Preprocessing Pipelines

Three distinct pipelines are implemented, each optimized for its model family.

| Step | ML Pipeline | DL Sequence Pipeline | Transformer Pipeline |
|---|---|---|---|
| Strip HTML tags | ✅ | ✅ | ✅ |
| Remove URLs | ✅ | ✅ | ✅ |
| Remove square brackets | ✅ | ✅ | ✅ |
| Expand contractions | ✅ | ✅ | ❌ |
| Lowercase | ✅ | ✅ | ❌ |
| Remove special chars | ✅ | ✅ | ❌ |
| Remove stopwords | ✅ (negations kept) | ❌ | ❌ |
| POS-aware lemmatization | ✅ | ✅ | ❌ |

**Key design decisions:**

- **Negation preservation** — words like *not*, *never*, *don't* are excluded from the stopword list in the ML pipeline. Removing them would flip sentiment meaning (e.g. "not good" → "good").
- **DL keeps context words** — RNN/LSTM rely on sequential context; removing stopwords breaks the positional signal the embedding layer learns.
- **Transformers get minimal cleaning** — DistilBERT's WordPiece tokenizer is trained on raw text and handles casing, punctuation, and sub-word morphology natively.

---

## Models

### Classical ML — TF-IDF + Classifiers

TF-IDF vectorization with unigrams + bigrams at two vocabulary sizes (10k and 50k features):

- **Logistic Regression** — strong, interpretable baseline
- **LinearSVC** — best classical ML performer; saved to disk
- **Multinomial Naive Bayes** — fastest to train; requires non-negative TF-IDF inputs

### Deep Learning — Sequence Models

```
Embedding (vocab=50k, dim=64) → [RNN | LSTM](64 units) → Dense(16/32, ReLU) → Dropout → Dense(1, Sigmoid)
```

- Max sequence length: **580 tokens** (covers ~95th percentile of review lengths)
- Optimizer: Adam; Loss: Binary Crossentropy
- Early stopping on `val_loss` with `patience=3`, restoring best weights

### Transformer — DistilBERT Fine-Tuning

Fine-tunes `distilbert-base-uncased` (66M parameters) for sequence classification:

```
[CLS] token → DistilBERT encoder (6 layers, 768-dim) → Classification head (2 labels)
```

| Hyperparameter | Value |
|---|---|
| Max token length | 256 |
| Epochs | 8 (early stopping, patience=2) |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Weight decay | 0.05 |
| Mixed precision (fp16) | ✅ (if GPU available) |

---

## Results

> Results reflect test-set performance on the 20% hold-out split.

| Model | TF-IDF Features | Accuracy | Notes |
|---|---|---|---|
| Logistic Regression | 10k | — | Solid baseline |
| Logistic Regression | 50k | — | Marginal improvement |
| LinearSVC | 10k | — | Best classical |
| LinearSVC | 50k | — | Best classical |
| Multinomial Naive Bayes | 10k | — | Fastest |
| Simple RNN | — | — | Prone to vanishing gradient |
| LSTM | — | — | Significantly better than RNN |
| **DistilBERT** | — | **~93%+** | Best overall |

> **Note:** Run the notebooks to populate exact accuracy/F1 scores from your environment. DistilBERT results depend on GPU availability and training duration.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU *(recommended for transformer fine-tuning)*

### 1. Clone the repository

```bash
git clone https://github.com/your-username/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
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

<details>
<summary>Core dependencies</summary>

```
pandas
numpy
scikit-learn
nltk
beautifulsoup4
wordcloud
matplotlib
seaborn
tensorflow
torch
transformers
datasets
kagglehub
joblib
```

</details>

### 4. Download NLTK assets

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
```

---

## Usage

### Run EDA

```bash
jupyter notebook eda.ipynb
```

### Run the full modeling pipeline

```bash
jupyter notebook sentiment_analysis_imdb_final.ipynb
```

Cells are ordered sequentially. Run all cells top-to-bottom. The dataset is downloaded automatically on first run.

### Load a saved model for inference

**LinearSVC (fastest)**

```python
import joblib

vectorizer = joblib.load("saved_models/ml/tfidf_50k_vectorizer.joblib")
model      = joblib.load("saved_models/ml/linear_svc_model.joblib")

review = ["This movie was an absolute masterpiece. Stunning performances."]
x = vectorizer.transform(review)
print(model.predict(x))  # [1] → positive
```

**LSTM**

```python
import joblib, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model     = load_model("saved_models/dl_lstm/lstm_sentiment_model.keras")
tokenizer = joblib.load("saved_models/dl_lstm/tokenizer_dl.joblib")
config    = json.load(open("saved_models/dl_lstm/dl_config.json"))

seq  = tokenizer.texts_to_sequences(["Terrible film, complete waste of time."])
padded = pad_sequences(seq, maxlen=config["max_length"], padding="post", truncating="post")
prob = model.predict(padded)[0][0]
print("positive" if prob > 0.5 else "negative")
```

**DistilBERT**

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="saved_models/transformer_distilbert",
    tokenizer="saved_models/transformer_distilbert"
)
print(classifier("One of the best films I've seen in years."))
```

---

## Saved Models

| Path | Contents | Use case |
|---|---|---|
| `saved_models/ml/` | LinearSVC + TF-IDF 50k | Fast CPU inference |
| `saved_models/dl_lstm/` | Keras LSTM + DL tokenizer + config JSON | Moderate accuracy, GPU optional |
| `saved_models/transformer_distilbert/` | DistilBERT weights + tokenizer | Highest accuracy, GPU recommended |

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy`, `kagglehub` |
| Text Processing | `nltk`, `beautifulsoup4`, `scikit-learn` |
| Classical ML | `scikit-learn` (LogisticRegression, LinearSVC, MultinomialNB) |
| Deep Learning | `tensorflow` / `keras` (Embedding, SimpleRNN, LSTM) |
| Transformers | `transformers`, `datasets`, `torch` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Model Persistence | `joblib` |

---

## License

This project is released under the [MIT License](LICENSE).

---

*Dataset credit: [Lakshmi Pathi — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)*
