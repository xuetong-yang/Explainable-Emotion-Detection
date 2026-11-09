import csv
import sys
import numpy as np
from feature_extraction import preprocess, tokenize, extract_features, vectorize, build_feature_map
import random
random.seed(123)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,classification_report


train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# Load the training data
with open(train_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)

# define valid labels
labels = {'joy', 'fear', 'shame', 'disgust', 'guilt', 'anger', 'sadness'}

# fixed label order for confusion matrix, matches the MLR and RoBERTa scripts
LABELS = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']

# filter valid training rows
valid_train_rows = []
for row in rows:
    if len(row) < 2:
        continue
    if row[0] not in labels:
        continue
    valid_train_rows.append(row)

# build label_map
label_map = {}
index = 0
for row in valid_train_rows:
    label = row[0]
    if label not in label_map:
        label_map[label] = index
        index += 1

# build feature_map using only valid training texts
texts = []
for row in valid_train_rows:
    texts.append(row[1])

feature_map = build_feature_map(texts)

# initialize weight
weight = {}
for label in label_map:
    weight[label] = np.zeros(len(feature_map))

# initialize weight_sum for averaging
weight_sum = {}
for label in label_map:
    weight_sum[label] = np.zeros(len(feature_map))

# define score = W · φ(x)
def score(weight_vector, sparse_vector):
    s = 0
    for i in sparse_vector:
        s += weight_vector[i]
    return s

# define predict
def predict(weight, sparse_vector):
    best_score = float('-inf')
    best_label = None
    for label in weight:
        s = score(weight[label], sparse_vector)
        if s > best_score:
            best_score = s
            best_label = label
    return best_label

# training loop
T = 15
for t in range(T):
    random.shuffle(valid_train_rows)
    for row in valid_train_rows:
        gold_label = row[0]
        text = row[1]
        vector = vectorize(text, feature_map)
        pred_label = predict(weight, vector)
        if pred_label != gold_label:
            for i in vector:
                weight[gold_label][i] += 1
                weight[pred_label][i] -= 1
        for label in weight:
            weight_sum[label] += weight[label]

# calculate average weights
total = T * len(valid_train_rows)
weight_avg = {}
for label in weight_sum:
    weight_avg[label] = weight_sum[label] / total

# predict and write output
with open(test_file, 'r', encoding='utf-8') as val:
    reader = csv.reader(val)
    rows = list(reader)

gold_labels = []
pred_labels = []

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    for row in rows:
        if len(row) < 2:
            writer.writerow(['', ''])  # keep alignment with gold file
            continue
        gold_label = row[0]
        text = row[1]
        vector = vectorize(text, feature_map)
        pred = predict(weight_avg, vector)
        writer.writerow([pred, text])

        # only keep rows with a valid gold label for evaluation
        if gold_label in labels:
            gold_labels.append(gold_label)
            pred_labels.append(pred)

# sanity check: should match the 0.5546 accuracy
print("Test Accuracy:", accuracy_score(gold_labels, pred_labels))
print("Test F1:", f1_score(gold_labels, pred_labels, average='macro'))
print(classification_report(gold_labels, pred_labels, labels=LABELS, digits=3))

# confusion matrix, using the same label order as the MLR / RoBERTa scripts
cm_test = confusion_matrix(gold_labels, pred_labels, labels=LABELS)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=LABELS)
disp_test.plot()
plt.tight_layout()
plt.savefig('confusion_matrix_AvgPerceptron_test.png', dpi=150, bbox_inches='tight')
plt.close()
