from __future__ import annotations

import pickle
import re
from pathlib import Path

from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
PIPELINE_PATH = MODEL_DIR / "spam_pipeline.pkl"
LEGACY_MODEL_PATH = MODEL_DIR / "spam_model.pkl"
LEGACY_VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifact() -> dict:
    if PIPELINE_PATH.exists():
        with PIPELINE_PATH.open("rb") as pipeline_file:
            return {"mode": "pipeline", "model": pickle.load(pipeline_file)}

    if LEGACY_MODEL_PATH.exists() and LEGACY_VECTORIZER_PATH.exists():
        with LEGACY_MODEL_PATH.open("rb") as model_file, LEGACY_VECTORIZER_PATH.open("rb") as vectorizer_file:
            return {
                "mode": "legacy",
                "model": pickle.load(model_file),
                "vectorizer": pickle.load(vectorizer_file),
            }

    raise FileNotFoundError(
        "Model files are missing. Run train_model.py to generate the trained artifacts in the model/ folder."
    )


try:
    MODEL_ARTIFACT = load_artifact()
    MODEL_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - handled in the UI
    MODEL_ARTIFACT = None
    MODEL_LOAD_ERROR = str(exc)


def predict_message(message: str) -> tuple[str, str, str]:
    if not message.strip():
        raise ValueError("Please enter a message to analyze.")

    if MODEL_ARTIFACT is None:
        raise RuntimeError(MODEL_LOAD_ERROR or "Model is unavailable.")

    if MODEL_ARTIFACT["mode"] == "pipeline":
        model = MODEL_ARTIFACT["model"]
        prediction = int(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
    else:
        cleaned_message = normalize_text(message)
        vector_input = MODEL_ARTIFACT["vectorizer"].transform([cleaned_message])
        model = MODEL_ARTIFACT["model"]
        prediction = int(model.predict(vector_input)[0])
        probabilities = model.predict_proba(vector_input)[0]

    confidence = probabilities[prediction]
    label = "Spam Email 🚫" if prediction == 1 else "Not Spam Email ✅"
    result_class = "spam" if prediction == 1 else "ham"
    confidence_text = f"Confidence: {confidence * 100:.1f}%"

    return label, confidence_text, result_class


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", error_text=MODEL_LOAD_ERROR)


@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message", "").strip()

    try:
        prediction_text, confidence_text, result_class = predict_message(message)
        return render_template(
            "index.html",
            prediction_text=prediction_text,
            confidence_text=confidence_text,
            result_class=result_class,
            message_value=message,
        )
    except Exception as exc:
        return render_template(
            "index.html",
            error_text=str(exc),
            message_value=message,
        )


if __name__ == "__main__":
    app.run(debug=False)