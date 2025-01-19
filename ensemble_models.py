import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json

# Download NLTK resources
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load mixed dataset for training and testing
print("Loading mixed dataset...")
mixed_df = pd.read_csv('datasets/mixed_dataset.csv', encoding='ISO-8859-1').dropna()

X = mixed_df['SMSes']
y = mixed_df['Labels']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess text
print("Preprocessing text...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text.lower())  # Remove non-alphanumeric characters
    text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(text) if word not in stop_words])
    return text

X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# TF-IDF Vectorization for NB, SVM, DT
print("Vectorizing text using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

# Function to calculate ASR
def calculate_asr(model, X_test, y_test):
    spam_indices = y_test == 1  # Assuming spam is labeled as 1
    X_test_spam = X_test[spam_indices]
    y_test_spam = y_test[spam_indices]

    if isinstance(model, Sequential):  # LSTM case
        X_test_spam = pad_sequences(tokenizer.texts_to_sequences(X_test_spam), maxlen=100)
        predictions = (model.predict(X_test_spam) > 0.5).astype("int32").flatten()
    else:  # Non-LSTM models
        X_test_spam = tfidf.transform(X_test_spam).toarray()
        predictions = model.predict(X_test_spam)

    misclassified_as_ham = (predictions == 0).sum()
    total_spam = len(y_test_spam)
    asr = misclassified_as_ham / total_spam if total_spam > 0 else 0
    return asr

# Initialize results storage
ensemble_results = {
    "Naive Bayes": {},
    "SVM": {},
    "Decision Tree": {},
    "LSTM": {},
    "Stacking Ensemble": {},
    "Voting Ensemble": {}
}

# Baseline Models
print("Training baseline models...")
nb = MultinomialNB()
svm = SVC(kernel='linear', probability=True)
dt = DecisionTreeClassifier()

print("Training Naive Bayes...")
nb.fit(X_train_tfidf, y_train)
print("Naive Bayes training completed.")

print("Training SVM...")
svm.fit(X_train_tfidf, y_train)
print("SVM training completed.")

print("Training Decision Tree...")
dt.fit(X_train_tfidf, y_train)
print("Decision Tree training completed.")

y_pred_nb = nb.predict(X_test_tfidf)
y_pred_svm = svm.predict(X_test_tfidf)
y_pred_dt = dt.predict(X_test_tfidf)

ensemble_results["Naive Bayes"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_nb, average="weighted"),
    "ASR": calculate_asr(nb, X_test, y_test)
}
ensemble_results["SVM"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_svm, average="weighted"),
    "ASR": calculate_asr(svm, X_test, y_test)
}
ensemble_results["Decision Tree"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_dt, average="weighted"),
    "ASR": calculate_asr(dt, X_test, y_test)
}

# Tokenization for LSTM
print("Tokenizing text for LSTM...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_len = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# LSTM Model
print("Building LSTM model...")
lstm_model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype("int32")
ensemble_results["LSTM"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_lstm, average="weighted"),
    "ASR": calculate_asr(lstm_model, X_test, y_test)
}

# Stacking Ensemble
print("Training Stacking Ensemble...")
stacking_model = StackingClassifier(
    estimators=[
        ('nb', nb),
        ('svm', svm),
        ('dt', dt)
    ],
    final_estimator=LogisticRegression()
)
stacking_model.fit(X_train_tfidf, y_train)
y_pred_stack = stacking_model.predict(X_test_tfidf)
ensemble_results["Stacking Ensemble"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_stack, average="weighted"),
    "ASR": calculate_asr(stacking_model, X_test, y_test)
}

# Voting Ensemble
print("Training Voting Ensemble...")
voting_model = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('svm', svm),
        ('dt', dt)
    ],
    voting='soft'  # Use soft voting for probability-based decisions
)
voting_model.fit(X_train_tfidf, y_train)
y_pred_vote = voting_model.predict(X_test_tfidf)
ensemble_results["Voting Ensemble"]["mixed"] = {
    "F1 Score": f1_score(y_test, y_pred_vote, average="weighted"),
    "ASR": calculate_asr(voting_model, X_test, y_test)
}

output_file = "evaluations/ensemble_results.json"
with open(output_file, "w") as f:
    json.dump(ensemble_results, f, indent=4)

print(f"Model evaluation results have been saved to {output_file}")