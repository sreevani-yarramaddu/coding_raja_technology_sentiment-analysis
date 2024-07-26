from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import webbrowser

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1', header=None)
data.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Data preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and links
    text = re.sub(r'http\S+|www\S+|[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

data['clean_text'] = data['text'].apply(preprocess_text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(data['clean_text'], data['target'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Model training
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Model evaluation
y_pred = nb_classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Deployment
def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    vectorized_text = tfidf_vectorizer.transform([cleaned_text])
    prediction = nb_classifier.predict(vectorized_text)
    if prediction == 0:
        return "Negative"
    elif prediction == 2:
        return "Neutral"
    else:
        return "Positive"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('index.html', sentiment=sentiment)

if __name__ == '__main__':
    # Open the browser automatically
    webbrowser.open('http://127.0.0.1:5000/')
    # Run the Flask app
    app.run(debug=True)
