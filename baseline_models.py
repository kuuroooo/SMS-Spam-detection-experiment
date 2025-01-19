import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('datasets/super_sms_dataset.csv', encoding='ISO-8859-1')
df.dropna(inplace=True)

# Split original dataset into train and test
print("Splitting original dataset...")
X = df['SMSes']
y = df['Labels']

# Initial split to create the test set
X_train_temp, X_test_original, y_train_temp, y_test_original = train_test_split(X, y, test_size=0.2, random_state=42)

# Load and process augmented datasets
print("Loading augmented datasets...")
augmented_files = [
    'datasets/magic_word_dataset.csv',
    'datasets/typos_dataset.csv',
    'datasets/synonym_dataset.csv',
    'datasets/spacing_dataset.csv',
]
methods = ["magic_word", "typos", "synonym", "spacing"]

augmented_test_sets = {}
for method, aug_file in zip(methods, augmented_files):
    aug_df = pd.read_csv(aug_file, encoding='ISO-8859-1')
    aug_df = aug_df.sample(n=2000, random_state=42)
    augmented_test_sets[method] = aug_df

# Tokenize and pad sequences
print("Tokenizing and padding sequences...")
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train_temp)
X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_temp), maxlen=100)

# TF-IDF Vectorization for SVM
print("Vectorizing text using TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train_temp).toarray()

# Initialize results dictionary
results = {
    "Naive Bayes": {},
    "SVM": {},
    "Decision Tree": {},
    "LSTM": {}
}

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


# Train and evaluate models
def train_and_evaluate():
    # Naive Bayes
    print("Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train_temp)
    for method, aug_df in augmented_test_sets.items():
        X_test_aug = aug_df['SMSes']
        y_test_aug = aug_df['Labels']

        # F1 Score
        X_test_aug_tfidf = tfidf.transform(X_test_aug).toarray()
        f1 = f1_score(y_test_aug, nb.predict(X_test_aug_tfidf), average="weighted")
        results["Naive Bayes"][method] = {
            "F1 Score": f1,
            "ASR": calculate_asr(nb, X_test_aug, y_test_aug),
            "Dataset": method
        }

    # Evaluate on original test set
    X_test_original_tfidf = tfidf.transform(X_test_original).toarray()
    f1 = f1_score(y_test_original, nb.predict(X_test_original_tfidf), average="weighted")
    results["Naive Bayes"]["original"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(nb, X_test_original, y_test_original),
        "Dataset": "original"
    }

    # Evaluate on mixed test set
    X_test_mixed = pd.concat([X_test_original] + [aug_df['SMSes'] for aug_df in augmented_test_sets.values()])
    y_test_mixed = pd.concat([y_test_original] + [aug_df['Labels'] for aug_df in augmented_test_sets.values()])
    X_test_mixed_tfidf = tfidf.transform(X_test_mixed).toarray()
    f1 = f1_score(y_test_mixed, nb.predict(X_test_mixed_tfidf), average="weighted")
    results["Naive Bayes"]["mixed"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(nb, X_test_mixed, y_test_mixed),
        "Dataset": "mixed"
    }

    # SVM
    print("Training SVM...")
    svm = SVC(kernel='linear')
    svm.fit(X_train_tfidf, y_train_temp)
    for method, aug_df in augmented_test_sets.items():
        X_test_aug = aug_df['SMSes']
        y_test_aug = aug_df['Labels']

        # F1 Score
        X_test_aug_tfidf = tfidf.transform(X_test_aug).toarray()
        f1 = f1_score(y_test_aug, svm.predict(X_test_aug_tfidf), average="weighted")
        results["SVM"][method] = {
            "F1 Score": f1,
            "ASR": calculate_asr(svm, X_test_aug, y_test_aug),
            "Dataset": method
        }

    # Evaluate on original test set
    f1 = f1_score(y_test_original, svm.predict(X_test_original_tfidf), average="weighted")
    results["SVM"]["original"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(svm, X_test_original, y_test_original),
        "Dataset": "original"
    }

    # Evaluate on mixed test set
    f1 = f1_score(y_test_mixed, svm.predict(X_test_mixed_tfidf), average="weighted")
    results["SVM"]["mixed"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(svm, X_test_mixed, y_test_mixed),
        "Dataset": "mixed"
    }

    # Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier()
    dt.fit(X_train_tfidf, y_train_temp)
    for method, aug_df in augmented_test_sets.items():
        X_test_aug = aug_df['SMSes']
        y_test_aug = aug_df['Labels']

        # F1 Score
        X_test_aug_tfidf = tfidf.transform(X_test_aug).toarray()
        f1 = f1_score(y_test_aug, dt.predict(X_test_aug_tfidf), average="weighted")
        results["Decision Tree"][method] = {
            "F1 Score": f1,
            "ASR": calculate_asr(dt, X_test_aug, y_test_aug),
            "Dataset": method
        }

    # Evaluate on original test set
    f1 = f1_score(y_test_original, dt.predict(X_test_original_tfidf), average="weighted")
    results["Decision Tree"]["original"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(dt, X_test_original, y_test_original),
        "Dataset": "original"
    }

    # Evaluate on mixed test set
    f1 = f1_score(y_test_mixed, dt.predict(X_test_mixed_tfidf), average="weighted")
    results["Decision Tree"]["mixed"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(dt, X_test_mixed, y_test_mixed),
        "Dataset": "mixed"
    }

    # LSTM
    print("Training LSTM...")
    lstm = Sequential([
        Embedding(input_dim=5000, output_dim=128),
        LSTM(64, return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_pad, y_train_temp, epochs=5, batch_size=32, validation_split=0.2)
    for method, aug_df in augmented_test_sets.items():
        X_test_aug = aug_df['SMSes']
        y_test_aug = aug_df['Labels']

        # F1 Score
        X_test_aug_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_aug), maxlen=100)
        y_pred = (lstm.predict(X_test_aug_pad) > 0.5).astype("int32").flatten()
        f1 = f1_score(y_test_aug, y_pred, average="weighted")
        results["LSTM"][method] = {
            "F1 Score": f1,
            "ASR": calculate_asr(lstm, X_test_aug, y_test_aug),
        }

    # Evaluate on original test set
    X_test_original_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_original), maxlen=100)
    y_pred = (lstm.predict(X_test_original_pad) > 0.5).astype("int32").flatten()
    f1 = f1_score(y_test_original, y_pred, average="weighted")
    results["LSTM"]["original"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(lstm, X_test_original, y_test_original),
        "Dataset": "original"
    }

    # Evaluate on mixed test set
    X_test_mixed_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_mixed), maxlen=100)
    y_pred = (lstm.predict(X_test_mixed_pad) > 0.5).astype("int32").flatten()
    f1 = f1_score(y_test_mixed, y_pred, average="weighted")
    results["LSTM"]["mixed"] = {
        "F1 Score": f1,
        "ASR": calculate_asr(lstm, X_test_mixed, y_test_mixed),
        "Dataset": "mixed"
    }

train_and_evaluate()

# Save results to file
with open("evaluations/baseline_results_with_asr.json", "w") as f:
    json.dump(results, f, indent=4)
print("Results saved to evaluations/baseline_results_with_asr.json")