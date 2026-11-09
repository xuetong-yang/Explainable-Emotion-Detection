import csv
import sys
import numpy as np
from feature_extraction import preprocess, tokenize, extract_features, vectorize, build_feature_map
import random

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

# Load the training data
with open(train_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    rows = list(reader)

# define valid labels
labels = {'joy', 'fear', 'shame', 'disgust', 'guilt', 'anger', 'sadness'}

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
        label_map[label] = index # assign a unique index to each label, which will be used to access the corresponding weight vector in the training loop
        index += 1
# print(label_map)
# label_map: {'joy': 0, 'fear': 1, 'shame': 2, 'disgust': 3, 'guilt': 4, 'anger': 5, 'sadness': 6}

# build feature_map using only valid training texts
texts = []
for row in valid_train_rows:
    texts.append(row[1])

feature_map = build_feature_map(texts)
# print(list(feature_map.items())[:4])
# feature_map: [('word=when', 0), ('word=i', 1), ('word=understood', 2), ('word=admitted', 3), ('word=university', 4)]

# initialize Weight
weight = {} # it is a dictionary because we want key(label)-value(weight vector) pairs.
for label in label_map:
    weight[label] = np.zeros(len(feature_map)) # A matrix of len(feature_map) consisting all zeros
# print(weight)

# define score = W · φ(x)
def score(weight_vector, sparse_vector): # weight_vector eg.weight["joy"]
    s = 0
    for i in sparse_vector: # iterate through the non-zero features in the sparse vector
        s += weight_vector[i] # only add weights for present features, non-present ones are 0 and ignored
    return s

# define predict: given a text, predict the label with the highest score
def predict(weight, sparse_vector): # weight here is the whole weight matrix.
    best_score = float('-inf') # to make sure any score can be larger than it
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
    print("iteration:", t+1)
    random.shuffle(valid_train_rows)
    for row in valid_train_rows:
        gold_label = row[0]
        text = row[1]
        vector = vectorize(text, feature_map)
        pred_label = predict(weight, vector)
        if pred_label != gold_label:
            for i in vector:
                weight[gold_label][i] += 1 # strengthen the gold_label
                weight[pred_label][i] -= 1 # weaken the wrong pred_label

# predict and write output
with open(test_file, 'r', encoding='utf-8') as val:
    reader = csv.reader(val)
    rows = list(reader)

with open(output_file, 'w', newline='', encoding='utf-8') as out:
    writer = csv.writer(out)
    for row in rows:
        if len(row) < 2:
            writer.writerow(['', ''])  # keep alignment with gold file
            continue
        text = row[1]
        vector = vectorize(text, feature_map)
        pred = predict(weight, vector)
        writer.writerow([pred, text])
