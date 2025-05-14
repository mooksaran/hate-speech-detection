import nltk
import random
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset (You should use a larger, labeled dataset for real applications)
data = [
    ("I hate you, you're the worst!", "hate_speech"),
    ("I love this place!", "non_hate_speech"),
    ("You're a terrible person.", "hate_speech"),
    ("Great job on your project!", "non_hate_speech"),
    # Add more labeled data here
]

# Preprocessing and feature extraction
def preprocess_text(text):
    # Tokenize
    words = word_tokenize(text)
    
    # Remove punctuation and convert to lowercase
    words = [word.lower() for word in words if word.isalnum()]
    
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words("english")]
    
    return " ".join(words)

# Prepare the dataset
random.shuffle(data)
texts, labels = zip(*data)
texts = [preprocess_text(text) for text in texts]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a simple Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
