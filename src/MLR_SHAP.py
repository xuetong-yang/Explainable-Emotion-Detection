import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shap_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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



train_texts, train_labels = load_data("data/isear-train.csv")
dev_texts, dev_labels = load_data("data/isear-val.csv")
test_texts, test_labels = load_data("data/isear-test.csv")


# TF-IDF Unigram + Bigram

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)  # change to (1,1) for unigram only
)
X_train = vectorizer.fit_transform(train_texts)
X_dev = vectorizer.transform(dev_texts)
X_test = vectorizer.transform(test_texts)

# Multinomial Logistic Regression (MLR)

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)
model.fit(X_train, train_labels)


# EVALUATION

dev_pred = model.predict(X_dev)
print("\n===== DEV SET =====")
print("Accuracy:", accuracy_score(dev_labels, dev_pred))
print(classification_report(dev_labels, dev_pred, digits=3))

test_pred = model.predict(X_test)
print("\n===== TEST SET =====")
print("Accuracy:", accuracy_score(test_labels, test_pred))
print(classification_report(test_labels, test_pred, digits=3))

# Dev confusion matrix
cm_dev = confusion_matrix(dev_labels, dev_pred, labels=model.classes_)
disp_dev = ConfusionMatrixDisplay(confusion_matrix=cm_dev, display_labels=model.classes_)
disp_dev.plot()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_MLR_dev.png'), bbox_inches="tight")
plt.close()

# Test confusion matrix
cm_test = confusion_matrix(test_labels, test_pred, labels=model.classes_)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=model.classes_)
disp_test.plot()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_MLR_test.png'), bbox_inches="tight")
plt.close()


# 1) TOP WORDS PER EMOTION — from model coefficients

feature_names = vectorizer.get_feature_names_out()

for i, label in enumerate(model.classes_):
    top_idx = model.coef_[i].argsort()[-10:]
    top_words = feature_names[top_idx]
    top_weights = model.coef_[i][top_idx]

    print(f"{label}: {top_words}")

    order = np.argsort(top_weights)
    plt.figure(figsize=(8, 5))
    plt.barh(top_words[order], top_weights[order], color="steelblue")
    plt.title(f"Top words — {label}")
    plt.xlabel("Logistic Regression coefficient")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"top_words_{label}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# 2) SHAP EXPLAINABILITY 

background_size = min(200, X_train.shape[0])
background_idx = np.random.RandomState(42).choice(
    X_train.shape[0], background_size, replace=False
)
background = X_train[background_idx]

explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

n_per_label = 7  

indices_by_label = defaultdict(list)
for idx, lbl in enumerate(test_labels):
    indices_by_label[lbl].append(idx)

sample_idx = []
for lbl, idxs in indices_by_label.items():
    sample_idx.extend(idxs[:n_per_label])

X_explain = X_test[sample_idx]
shap_values = explainer.shap_values(X_explain)


def get_class_shap(shap_values, class_idx):
    if isinstance(shap_values, list):
        return shap_values[class_idx]
    else:
        return shap_values[:, :, class_idx]



# TOP TOKENS DRIVING EACH EMOTION 

X_explain_dense = X_explain.toarray()

for i, label in enumerate(model.classes_):
    class_shap = get_class_shap(shap_values, i)
    mean_shap = class_shap.mean(axis=0)

    top_n = 10
    top_idx = np.argsort(np.abs(mean_shap))[-top_n:][::-1]

    explanation = shap.Explanation(
        values=mean_shap[top_idx],
        base_values=explainer.expected_value[i],
        data=X_explain_dense.mean(axis=0)[top_idx],
        feature_names=feature_names[top_idx],
    )
    shap.plots.bar(explanation, max_display=top_n, show=False)
    fig = plt.gcf()  
    ax = plt.gca()
    ax.set_xlim(-0.4, 0.4)
    fig.suptitle(f"Top tokens driving '{label}' predictions")
    save_path = os.path.join(OUTPUT_DIR, f"shap_bar_{label}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# Example sentence explanations — one per emotion
sample_texts = [test_texts[i] for i in sample_idx]
sample_labels = [test_labels[i] for i in sample_idx]
sample_pred = model.predict(X_explain)

example_indices = {}
for idx, true_label in enumerate(sample_labels):
    if true_label not in example_indices:
        example_indices[true_label] = idx

print("\n===== SHAP EXAMPLE EXPLANATIONS =====")
for true_label, idx in example_indices.items():
    predicted_label = sample_pred[idx]
    class_idx = list(model.classes_).index(predicted_label)

    print(f"\nSentence: \"{sample_texts[idx]}\"")
    print(f"True label: {true_label} | Predicted: {predicted_label}")

    class_shap = get_class_shap(shap_values, class_idx)
    sv = class_shap[idx]
    x_row = X_explain_dense[idx]

    nonzero_idx = np.nonzero(x_row)[0]
    contributions = sorted(
        [(feature_names[j], sv[j]) for j in nonzero_idx],
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:8]

    print(f"Top contributing words toward '{predicted_label}':")
    for word, val in contributions:
        direction = "+" if val > 0 else "-"
        print(f"   {direction} {word}: {val:.4f}")

    shap.plots.waterfall(
        shap.Explanation(
            values=sv,
            base_values=explainer.expected_value[class_idx],
            data=x_row,
            feature_names=feature_names,
        ),
        max_display=10,
        show=False,
    )
    fig = plt.gcf()
    fig.suptitle(f"SHAP waterfall — \"{true_label}\" example, predicted '{predicted_label}'")
    save_path = os.path.join(OUTPUT_DIR, f"shap_waterfall_{true_label}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

print(f"\nAll visualizations saved to: {OUTPUT_DIR}")
