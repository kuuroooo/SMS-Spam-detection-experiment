import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

print("Loading mixed dataset...")
mixed_df = pd.read_csv('datasets/mixed_dataset.csv', encoding='ISO-8859-1').dropna()

X = mixed_df['SMSes']
y = mixed_df['Labels']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Shared tokenizer and TF-IDF vectorizer
print("Tokenizing and vectorizing...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=100)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=100)

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
defense_results = {model: {} for model in ["Naive Bayes", "SVM", "Decision Tree", "LSTM"]}

# Train and evaluate models
print("Training models...")
# Naive Bayes
print("Training Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
f1 = f1_score(y_test, nb.predict(X_test_tfidf), average="weighted")
asr = calculate_asr(nb, X_test, y_test)
defense_results["Naive Bayes"]["mixed"] = {"F1 Score": f1, "ASR": asr}

# SVM
try:
    print("Training SVM...")
    svm = SVC(kernel='linear', verbose=True)
    svm.fit(X_train_tfidf, y_train)
    print("SVM training completed.")
    f1 = f1_score(y_test, svm.predict(X_test_tfidf), average="weighted")
    asr = calculate_asr(svm, X_test, y_test)
    defense_results["SVM"]["mixed"] = {"F1 Score": f1, "ASR": asr}
except Exception as e:
    print(f"An error occurred during SVM training or evaluation: {e}")

# Decision Tree
print("Training Decision Tree...")
dt = DecisionTreeClassifier()
dt.fit(X_train_tfidf, y_train)
f1 = f1_score(y_test, dt.predict(X_test_tfidf), average="weighted")
asr = calculate_asr(dt, X_test, y_test)
defense_results["Decision Tree"]["mixed"] = {"F1 Score": f1, "ASR": asr}

# LSTM
print("Training LSTM...")
lstm = Sequential([
    Embedding(input_dim=5000, output_dim=128),
    LSTM(64, return_sequences=False),
    Dense(1, activation='sigmoid')
])
lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=False)
y_pred = (lstm.predict(X_test_pad) > 0.5).astype("int32").flatten()
f1 = f1_score(y_test, y_pred, average="weighted")
asr = calculate_asr(lstm, X_test, y_test)
defense_results["LSTM"]["mixed"] = {"F1 Score": f1, "ASR": asr}

# Save defense results
with open("evaluations/defense_results.json", "w") as f:
    json.dump(defense_results, f, indent=4)
print("Defense results saved.")