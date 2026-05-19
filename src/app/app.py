import os
import re
import json
import io
import sys
import types
import zipfile
import traceback
from pathlib import Path

import joblib
import numpy as np
import gradio as gr
from bs4 import BeautifulSoup

# =========================
# NLTK preprocessing
# =========================
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources():
    """
    Downloads required NLTK resources if missing.
    This matches the notebook preprocessing.
    """
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]

    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_resources()

lemmatizer = WordNetLemmatizer()
toktok_tokenizer = ToktokTokenizer()

contractions_dict = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "they'd": "they would",
}

stop_words = set(stopwords.words("english"))

# Words that reverse sentiment must not be filtered out.
negation_words = {
    "no", "nor", "not", "never",
    "don't", "doesn't", "didn't",
    "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't",
    "won't", "wouldn't", "shouldn't", "couldn't",
    "cannot", "can't", "n't",
}

stop_words = stop_words - negation_words


def remove_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text()


def remove_square_brackets(text):
    return re.sub(r"\[[^\]]*\]", "", str(text))


def remove_urls(text):
    return re.sub(r"http\S+|www\S+|https\S+", "", str(text))


def expand_contractions(text):
    return " ".join(contractions_dict.get(word, word) for word in str(text).split())


def remove_special_chars(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def pos_lemmatizer(tokens):
    pos_tags = nltk.pos_tag(tokens)
    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]


def preprocess_review_ml(text):
    """
    Same as notebook:
    aggressive preprocessing for TF-IDF + LinearSVC.
    """
    text = remove_html(text)
    text = remove_square_brackets(text)
    text = remove_urls(text)
    text = text.lower()
    text = expand_contractions(text)
    text = remove_special_chars(text)

    tokens = toktok_tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = pos_lemmatizer(tokens)

    return " ".join(tokens)


def preprocess_review_dl(text):
    """
    Same as notebook:
    light preprocessing for LSTM sequence model.
    Stop words are kept.
    """
    text = remove_html(text)
    text = remove_square_brackets(text)
    text = remove_urls(text)
    text = text.lower()
    text = expand_contractions(text)
    text = remove_special_chars(text)

    tokens = toktok_tokenizer.tokenize(text)
    tokens = pos_lemmatizer(tokens)

    return " ".join(tokens)


def preprocess_review_transformer(text):
    """
    Same as notebook:
    minimal preprocessing for DistilBERT.
    Keeps casing/contractions/stopwords.
    """
    text = remove_html(text)
    text = remove_square_brackets(text)
    text = remove_urls(text)
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


# =========================
# TensorFlow / Keras
# =========================
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass
except Exception:
    tf = None
    load_model = None
    pad_sequences = None


# =========================
# PyTorch / Transformers
# =========================
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    BertTokenizerFast = None


# =========================
# Paths
# =========================
APP_DIR = Path(__file__).resolve().parent
SRC_DIR = APP_DIR.parent
MODEL_ROOT = SRC_DIR / "saved_models"

ML_MODEL_PATH = MODEL_ROOT / "ml" / "linear_svc_model.joblib"
TFIDF_PATH = MODEL_ROOT / "ml" / "tfidf_50k_vectorizer.joblib"

LSTM_MODEL_PATH = MODEL_ROOT / "dl_lstm" / "lstm_sentiment_model.keras"
LSTM_TOKENIZER_PATH = MODEL_ROOT / "dl_lstm" / "tokenizer_dl.joblib"
LSTM_CONFIG_PATH = MODEL_ROOT / "dl_lstm" / "dl_config.json"

TRANSFORMER_PATH = MODEL_ROOT / "transformer_distilbert"
TRANSFORMER_TOKENIZER_FILE = TRANSFORMER_PATH / "tokenizer_bert.json"
TRANSFORMER_TOKENIZER_CONFIG = TRANSFORMER_PATH / "tokenizer_bert_config.json"

LABEL_MAP = {
    0: "Negative",
    1: "Positive",
}


# =========================
# Load models safely
# =========================
MODEL_STATUS = {}

ml_model = None
tfidf_vectorizer = None
lstm_model = None
tokenizer_dl = None
dl_config = None
transformer_tokenizer = None
transformer_model = None
device = None


def load_resource(name, loader):
    try:
        value = loader()
        MODEL_STATUS[name] = "Loaded"
        return value
    except Exception as exc:
        MODEL_STATUS[name] = f"Not loaded: {exc}"
        return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def install_keras_pickle_aliases():
    """
    Tokenizers saved with Keras 3 may reference keras.src.* modules.
    TensorFlow/Keras 2.10 keeps the same Tokenizer class in keras.preprocessing.*.
    """
    try:
        import keras.preprocessing.text as keras_text
        import keras.preprocessing.sequence as keras_sequence
    except Exception:
        return

    for module_name in [
        "keras.src",
        "keras.src.legacy",
        "keras.src.legacy.preprocessing",
        "keras.src.preprocessing",
    ]:
        sys.modules.setdefault(module_name, types.ModuleType(module_name))

    sys.modules["keras.src.legacy.preprocessing.text"] = keras_text
    sys.modules["keras.src.legacy.preprocessing.sequence"] = keras_sequence
    sys.modules["keras.src.preprocessing.text"] = keras_text
    sys.modules["keras.src.preprocessing.sequence"] = keras_sequence


def load_joblib_with_keras_aliases(path):
    install_keras_pickle_aliases()
    return joblib.load(path)


def keras_dtype_name(dtype_config):
    if isinstance(dtype_config, dict):
        return dtype_config.get("config", {}).get("name", "float32")
    return dtype_config or "float32"


def load_keras3_lstm_archive(path):
    """
    Load the Keras 3 .keras zip format in the older TensorFlow/Keras 2.10 env.
    The saved model is a simple Sequential Embedding/LSTM/Dense stack.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is not installed.")

    import h5py

    with zipfile.ZipFile(path) as archive:
        model_config = json.loads(archive.read("config.json"))
        weights_bytes = archive.read("model.weights.h5")

    layers_config = model_config["config"]["layers"]
    build_shape = model_config.get("build_config", {}).get("input_shape")
    max_length = int(build_shape[-1]) if build_shape else 580

    keras_layers = []
    for layer in layers_config:
        class_name = layer["class_name"]
        config = layer["config"]

        if class_name == "InputLayer":
            continue

        common = {
            "name": config.get("name"),
            "trainable": config.get("trainable", True),
            "dtype": keras_dtype_name(config.get("dtype")),
        }

        if class_name == "Embedding":
            keras_layers.append(
                tf.keras.layers.Embedding(
                    input_dim=config["input_dim"],
                    output_dim=config["output_dim"],
                    mask_zero=config.get("mask_zero", False),
                    input_length=max_length,
                    **common,
                )
            )
        elif class_name == "LSTM":
            keras_layers.append(
                tf.keras.layers.LSTM(
                    units=config["units"],
                    activation=config.get("activation", "tanh"),
                    recurrent_activation=config.get("recurrent_activation", "sigmoid"),
                    use_bias=config.get("use_bias", True),
                    return_sequences=config.get("return_sequences", False),
                    return_state=config.get("return_state", False),
                    go_backwards=config.get("go_backwards", False),
                    stateful=config.get("stateful", False),
                    unroll=config.get("unroll", False),
                    dropout=config.get("dropout", 0.0),
                    recurrent_dropout=config.get("recurrent_dropout", 0.0),
                    **common,
                )
            )
        elif class_name == "Dense":
            keras_layers.append(
                tf.keras.layers.Dense(
                    units=config["units"],
                    activation=config.get("activation"),
                    use_bias=config.get("use_bias", True),
                    **common,
                )
            )
        elif class_name == "Dropout":
            keras_layers.append(
                tf.keras.layers.Dropout(
                    rate=config["rate"],
                    **common,
                )
            )
        else:
            raise ValueError(f"Unsupported Keras 3 layer in fallback loader: {class_name}")

    model = tf.keras.Sequential(keras_layers, name=model_config["config"].get("name"))
    model.build((None, max_length))

    class_counts = {}
    with h5py.File(io.BytesIO(weights_bytes), "r") as weights_file:
        h5_layers = weights_file["layers"]

        for model_layer, layer_config in zip(model.layers, [l for l in layers_config if l["class_name"] != "InputLayer"]):
            class_name = layer_config["class_name"]
            base_name = class_name.lower()
            occurrence = class_counts.get(base_name, 0)
            class_counts[base_name] = occurrence + 1
            h5_name = base_name if occurrence == 0 else f"{base_name}_{occurrence}"

            if h5_name not in h5_layers:
                continue

            group = h5_layers[h5_name]
            if class_name == "Embedding":
                model_layer.set_weights([group["vars"]["0"][()]])
            elif class_name == "LSTM":
                vars_group = group["cell"]["vars"]
                model_layer.set_weights([
                    vars_group["0"][()],
                    vars_group["1"][()],
                    vars_group["2"][()],
                ])
            elif class_name == "Dense":
                model_layer.set_weights([
                    group["vars"]["0"][()],
                    group["vars"]["1"][()],
                ])

    return model


def load_lstm_model(path):
    try:
        return load_model(path, compile=False)
    except Exception:
        if zipfile.is_zipfile(path):
            return load_keras3_lstm_archive(path)
        raise


def load_transformer_tokenizer():
    if (TRANSFORMER_PATH / "tokenizer.json").exists():
        return AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))

    if TRANSFORMER_TOKENIZER_FILE.exists() and BertTokenizerFast is not None:
        tokenizer_kwargs = {}
        if TRANSFORMER_TOKENIZER_CONFIG.exists():
            tokenizer_kwargs = load_json(TRANSFORMER_TOKENIZER_CONFIG)

        return BertTokenizerFast(
            tokenizer_file=str(TRANSFORMER_TOKENIZER_FILE),
            **tokenizer_kwargs,
        )

    return AutoTokenizer.from_pretrained(str(TRANSFORMER_PATH))


ml_model = load_resource(
    "ML LinearSVC",
    lambda: joblib.load(ML_MODEL_PATH),
)

tfidf_vectorizer = load_resource(
    "TF-IDF Vectorizer",
    lambda: joblib.load(TFIDF_PATH),
)

if load_model is not None:
    lstm_model = load_resource(
        "LSTM Model",
        lambda: load_lstm_model(LSTM_MODEL_PATH),
    )
else:
    MODEL_STATUS["LSTM Model"] = "TensorFlow is not installed."

tokenizer_dl = load_resource(
    "LSTM Tokenizer",
    lambda: load_joblib_with_keras_aliases(LSTM_TOKENIZER_PATH),
)

dl_config = load_resource(
    "LSTM Config",
    lambda: load_json(LSTM_CONFIG_PATH),
)

if torch is not None and AutoTokenizer is not None and AutoModelForSequenceClassification is not None:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = torch.device("cpu")

    transformer_tokenizer = load_resource(
        "Transformer Tokenizer",
        load_transformer_tokenizer,
    )

    transformer_model = load_resource(
        "Transformer Model",
        lambda: AutoModelForSequenceClassification.from_pretrained(str(TRANSFORMER_PATH)),
    )

    if transformer_model is not None:
        transformer_model.to(device)
        transformer_model.eval()
else:
    MODEL_STATUS["Transformer"] = "PyTorch/Transformers is not installed."


# =========================
# Prediction helpers
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(logits):
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()


def unavailable_row(model_name, reason):
    return [
        model_name,
        "Unavailable",
        "N/A",
        reason,
    ]


def predict_ml(text):
    if ml_model is None or tfidf_vectorizer is None:
        return unavailable_row(
            "LinearSVC + TF-IDF",
            f"{MODEL_STATUS.get('ML LinearSVC')} | {MODEL_STATUS.get('TF-IDF Vectorizer')}",
        )

    processed = preprocess_review_ml(text)

    if not processed:
        return ["LinearSVC + TF-IDF", "Empty", "N/A", "No valid tokens after preprocessing."]

    X = tfidf_vectorizer.transform([processed])
    pred = int(ml_model.predict(X)[0])

    # LinearSVC does not output real probabilities.
    if hasattr(ml_model, "decision_function"):
        margin = ml_model.decision_function(X)
        score = float(margin[0]) if np.ndim(margin) == 1 else float(margin[0][pred])
        approx_pos_conf = float(sigmoid(score))
        confidence = approx_pos_conf if pred == 1 else 1 - approx_pos_conf
        details = f"Approx confidence from decision margin: {score:.4f}. Not calibrated probability."
    else:
        confidence = None
        details = "No confidence score available."

    return [
        "LinearSVC + TF-IDF",
        LABEL_MAP.get(pred, str(pred)),
        "N/A" if confidence is None else f"{confidence:.4f}",
        details,
    ]


def predict_lstm(text):
    if lstm_model is None or tokenizer_dl is None or dl_config is None or pad_sequences is None:
        return unavailable_row(
            "LSTM",
            f"{MODEL_STATUS.get('LSTM Model')} | {MODEL_STATUS.get('LSTM Tokenizer')} | {MODEL_STATUS.get('LSTM Config')}",
        )

    processed = preprocess_review_dl(text)

    if not processed:
        return ["LSTM", "Empty", "N/A", "No valid tokens after preprocessing."]

    max_length = int(dl_config.get("max_length", 580))

    seq = tokenizer_dl.texts_to_sequences([processed])
    padded = pad_sequences(
        seq,
        maxlen=max_length,
        padding="post",
        truncating="post",
    )

    prob_positive = float(lstm_model.predict(padded, verbose=0)[0][0])
    pred = 1 if prob_positive >= 0.5 else 0
    confidence = prob_positive if pred == 1 else 1 - prob_positive

    return [
        "LSTM",
        LABEL_MAP[pred],
        f"{confidence:.4f}",
        f"Positive probability: {prob_positive:.4f}",
    ]


def predict_transformer(text):
    if transformer_model is None or transformer_tokenizer is None or torch is None:
        return unavailable_row(
            "DistilBERT",
            f"{MODEL_STATUS.get('Transformer Model')} | {MODEL_STATUS.get('Transformer Tokenizer')}",
        )

    processed = preprocess_review_transformer(text)

    if not processed:
        return ["DistilBERT", "Empty", "N/A", "Empty text after preprocessing."]

    # Notebook tokenizer max_length was 256.
    inputs = transformer_tokenizer(
        [processed],
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    # The saved tokenizer can return token_type_ids, but DistilBERT does not use them.
    accepted_inputs = set(transformer_model.forward.__code__.co_varnames)
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
        if k in accepted_inputs
    }

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]

    probs = softmax(logits)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])

    return [
        "DistilBERT",
        LABEL_MAP.get(pred, str(pred)),
        f"{confidence:.4f}",
        f"Negative probability: {float(probs[0]):.4f} | Positive probability: {float(probs[1]):.4f}",
    ]


def predict_all(review):
    review = "" if review is None else str(review).strip()

    if not review:
        return [], "Enter a review first.", build_status_text()

    rows = [
        predict_ml(review),
        predict_lstm(review),
        predict_transformer(review),
    ]

    valid_predictions = [
        row[1] for row in rows
        if row[1] in {"Positive", "Negative"}
    ]

    positive_votes = valid_predictions.count("Positive")
    negative_votes = valid_predictions.count("Negative")

    if positive_votes > negative_votes:
        final_decision = "Positive"
    elif negative_votes > positive_votes:
        final_decision = "Negative"
    elif positive_votes == negative_votes and positive_votes > 0:
        final_decision = "Tie"
    else:
        final_decision = "No available model"

    summary = (
        f"### Final Decision: **{final_decision}**\n\n"
        f"- Positive votes: **{positive_votes}**\n"
        f"- Negative votes: **{negative_votes}**"
    )

    return rows, summary, build_status_text()


def build_status_text():
    lines = ["### Model loading status"]
    for name, status in MODEL_STATUS.items():
        lines.append(f"- **{name}:** {status}")
    return "\n".join(lines)


# =========================
# Gradio UI
# =========================
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
"""

with gr.Blocks(title="IMDB Sentiment Analysis - 3 Models", css=custom_css) as demo:
    gr.Markdown(
        """
# IMDB Sentiment Analysis Deployment

This app uses the same preprocessing logic extracted from your notebook:

- **ML:** → LinearSVC  
- **DL:** → LSTM  
- **Transformer:** → DistilBERT tokenizer/model  
"""
    )

    with gr.Row():
        review_input = gr.Textbox(
            label="Movie Review",
            placeholder="Type a movie review here...",
            lines=7,
        )

    analyze_btn = gr.Button("Analyze Sentiment", variant="primary")

    output_table = gr.Dataframe(
        headers=["Model", "Prediction", "Confidence", "Details"],
        label="Predictions",
        wrap=True,
        interactive=False,
    )

    final_output = gr.Markdown()
    status_output = gr.Markdown(value=build_status_text())

    gr.Examples(
        examples=[
            ["This movie was amazing. The acting was great and the story was very emotional."],
            ["Very boring movie. Bad acting and weak story. I do not recommend it."],
            ["The movie was okay, not great but not terrible either."],
        ],
        inputs=review_input,
    )

    analyze_btn.click(
        fn=predict_all,
        inputs=review_input,
        outputs=[output_table, final_output, status_output],
    )


if __name__ == "__main__":
    demo.launch()
