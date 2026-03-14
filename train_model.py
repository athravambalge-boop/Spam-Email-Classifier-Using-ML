import pandas as pd
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

ps = PorterStemmer()

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


df = pd.read_csv("dataset/spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({'ham':0,'spam':1})

df['transformed_text'] = df['message'].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=3000)

X = vectorizer.fit_transform(df['transformed_text']).toarray()

y = df['label'].values

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=2
)

model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))

pickle.dump(vectorizer,open('model/vectorizer.pkl','wb'))
pickle.dump(model,open('model/spam_model.pkl','wb'))