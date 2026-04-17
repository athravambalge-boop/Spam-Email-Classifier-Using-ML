from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "spam.csv"
MODEL_DIR = BASE_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "spam_pipeline.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "spam_model.pkl"
LEGACY_VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    classifier = MultinomialNB(alpha=0.5)

    return Pipeline(
        [
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATASET_PATH, encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_samples": int(len(df)),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    with PIPELINE_PATH.open("wb") as pipeline_file:
        pickle.dump(pipeline, pipeline_file)

    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]

    with LEGACY_VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    with LEGACY_MODEL_PATH.open("wb") as model_file:
        pickle.dump(classifier, model_file)


if __name__ == "__main__":
    main()