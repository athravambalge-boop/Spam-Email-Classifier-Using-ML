from flask import Flask, render_template, request
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

ps = PorterStemmer()

# Load trained model and vectorizer
model = pickle.load(open('model/spam_model.pkl','rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl','rb'))


def clean_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    transformed = clean_text(message)

    vector_input = vectorizer.transform([transformed])

    result = model.predict(vector_input)[0]

    if result == 1:
        prediction = "Spam Email 🚫"
    else:
        prediction = "Not Spam Email ✅"

    return render_template('index.html', prediction_text=prediction)


if __name__ == "__main__":
    app.run(debug=True)