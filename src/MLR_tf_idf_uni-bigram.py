import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def load_data(file_path):
    texts = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        valid_labels = {'joy', 'fear', 'shame', 'disgust', 'guilt', 'anger', 'sadness'}

        for row in reader:
            if len(row) < 2:
                continue
            if row[0] not in valid_labels:
                continue

            labels.append(row[0])
            texts.append(row[1])

    return texts, labels


# Load data
train_texts, train_labels = load_data("data/isear-train.csv")
dev_texts, dev_labels = load_data("data/isear-val.csv")
test_texts, test_labels = load_data("data/isear-test.csv")


# TF-IDF Unigram + Bigram
vectorizer = TfidfVectorizer(
   ngram_range=(1, 2) # change to (1,1) for unigram only
)

X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

# Multinomial Logistic Regression (MLR)
# A probabilistic classifier that:
# - assigns probabilities to each emotion class
# - selects class with highest probability

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

# Train model on TF-IDF features
model.fit(X_train, train_labels)

# EVALUATION
# Accuracy → overall correctness
# Classification report → precision, recall, F1 per emotion

dev_pred = model.predict(X_dev)

print("\n===== DEV SET =====")
print("Accuracy:", accuracy_score(dev_labels, dev_pred,))
print(classification_report(dev_labels, dev_pred, digits=3))

test_pred = model.predict(X_test)

print("\n===== TEST SET =====")
print("Accuracy:", accuracy_score(test_labels, test_pred,))
print(classification_report(test_labels, test_pred, digits=3))

# Dev confusion matrix
cm_dev = confusion_matrix(dev_labels, dev_pred, labels=model.classes_)
disp_dev = ConfusionMatrixDisplay(confusion_matrix=cm_dev, display_labels=model.classes_)
disp_dev.plot()
plt.savefig('confusion_matrix_MLR_dev.png')

# Test confusion matrix
cm_test = confusion_matrix(test_labels, test_pred, labels=model.classes_)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=model.classes_)
disp_test.plot()
plt.savefig('confusion_matrix_MLR_test.png')

# Weight Analysis
feature_names = vectorizer.get_feature_names_out()
for i, label in enumerate(model.classes_):
    top_features = feature_names[model.coef_[i].argsort()[-10:]]
    print(f"{label}: {top_features}")
