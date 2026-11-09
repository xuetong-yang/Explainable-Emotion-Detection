import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap

OUTPUT_DIR = 'mlr_shap_outputs'
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


train_file = sys.argv[1]
dev_file = sys.argv[2]
test_file = sys.argv[3]

train_texts, train_labels = load_data(train_file)
dev_texts, dev_labels = load_data(dev_file)
test_texts, test_labels = load_data(test_file)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
model.fit(X_train, train_labels)

feature_names = vectorizer.get_feature_names_out()

background_size = min(200, X_train.shape[0])
background_idx = np.random.RandomState(42).choice(X_train.shape[0], background_size, replace=False)
background = X_train[background_idx]
explainer = shap.LinearExplainer(model, background, feature_perturbation="interventional")

target_sentences = [
    (
        "We were at secondary school and I was making a lot of noise.  The teacher came to ask us as to who was making all the noise.  I did not own up so the  whole class was punished.",
        "guilt",
        "guilt_anger"
    ),
    (
        "I had this feeling before my graduation and never dared to speak to my relatives and  friends.  Because of this feeling I would avoid situations, however I do not know the reason for this.",
        "shame",
        "shame_guilt_both_wrong"
    ),
]

print("\n===== SPECIFIC EXAMPLE WATERFALL CHARTS =====")

TOP_N = 10  # number of features to show in waterfall

for text, true_label, desc in target_sentences:
    matched_idx = None
    for idx, t in enumerate(test_texts):
        if text.strip()[:50] in t:
            matched_idx = idx
            break

    if matched_idx is None:
        print(f"WARNING: Could not find in test set, vectorizing directly: {text[:80]}...")
        x_vec = vectorizer.transform([text])
    else:
        x_vec = X_test[matched_idx:matched_idx+1]

    predicted_label = model.predict(x_vec)[0]
    class_idx = list(model.classes_).index(predicted_label)

    sv = explainer.shap_values(x_vec)
    if isinstance(sv, list):
        sv_class = sv[class_idx][0]
    else:
        sv_class = sv[0, :, class_idx]

    x_dense = x_vec.toarray()[0]

    # Only keep top N features by absolute SHAP value
    # This removes the "other features" row entirely
    top_idx = np.argsort(np.abs(sv_class))[-TOP_N:]
    sv_top = sv_class[top_idx]
    x_top = x_dense[top_idx]
    fn_top = feature_names[top_idx]

    print(f"\n[{desc}]")
    print(f"Text: {text[:100]}...")
    print(f"True: {true_label} | MLR predicted: {predicted_label}")

    explanation = shap.Explanation(
        values=sv_top,
        base_values=explainer.expected_value[class_idx],
        data=x_top,
        feature_names=fn_top,
    )

    shap.plots.waterfall(explanation, max_display=TOP_N, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP waterfall (MLR) — true: '{true_label}', predicted: '{predicted_label}'\n{text[:80]}...",
        fontsize=9
    )
    save_path = os.path.join(OUTPUT_DIR, f"mlr_shap_waterfall_specific_{desc}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
