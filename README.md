# Spam Email Classifier

An end-to-end machine learning web app that classifies message text as spam or legitimate mail. The project is designed to look stronger on a resume because it includes data preparation, model training, evaluation metrics, and a deployed Flask interface.


## Tech Stack

- Python
- Flask
- pandas
- scikit-learn
- joblib / pickle

## Project Structure

```text
Spam Email Classifier/
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- README.md
|-- dataset/
|   `-- spam.csv
|-- model/
|   |-- spam_pipeline.pkl
|   |-- spam_model.pkl
|   |-- vectorizer.pkl
|   `-- metrics.json
|-- templates/
|   `-- index.html
`-- static/
    `-- style.css
```

## How It Works

1. The training script loads the spam dataset and splits it into train/test sets.
2. A TF-IDF vectorizer converts raw text into numeric features.
3. A Multinomial Naive Bayes classifier learns from those features.
4. The app uses the trained artifact to return a prediction and confidence score.

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

## Train the Model

Run the training script to generate the model files and metrics:

```powershell
python train_model.py
```

This creates or updates:

- `model/spam_pipeline.pkl`
- `model/spam_model.pkl`
- `model/vectorizer.pkl`
- `model/metrics.json`

The script also prints accuracy, precision, recall, F1 score, and the confusion matrix.

## Run the Web App

```powershell
python app.py
```

Open:

`http://127.0.0.1:5000`


## Troubleshooting

- `FileNotFoundError` for model files: run `python train_model.py` first.
- App starts but shows a model error: confirm the `model/` directory contains the generated pickle files.
- Wrong working directory: run commands from the folder that contains `app.py`.

## Author
ATHARVA PRASHANT AMBALGE

## License

This project is for educational use.
