import csv

# STEP 1: IMPLEMENTATION 

def load_data(file_path): 
    texts = []
    labels = []
# We expect [label, text] format in each row

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
# Some rows may be broken or incomplete, we skip them
            label = row[0]
            text = row[1]

            texts.append(text)
            labels.append(label)

    return texts, labels

#texts, labels = load_data("isear-train.csv")

#print(len(texts), len(labels))
#print(texts[0])
#print(labels[0])

# STEP 2: PREPROCESSING
# Normalizing the text

def preprocess(text):
    # lowercase 
    text = text.lower()

    # remove quotes
    text = text.replace('"', '')

    # remove punctuation
    punctuation = '.,!?;:()[]{}'
    for p in punctuation:
        text = text.replace(p, '')

    # We want to ignore defined stopwords (common words) that are not related to emotions
    stopwords = {
        "a", "an", "the", "is", "in", "on", "at", "and", "or", "of",
        "to", "for", "with", "as", "by", "this", "that", "it", "be",
        "are", "was", "were", "am",
    }

    # split sentence into words
    words = text.split()

    # remove stopwords
    filtered_words = []
    for word in words:
        if word not in stopwords:
            filtered_words.append(word)

    text = " ".join(filtered_words)

    return text

# STEP 3: TOKENIZATION
def tokenize(text):
    return text.split()

# STEP 4: FEATURE EXTRACTION 
# Each word becomes a feature
# Goal here is to convert tokens into features usable by our model
def extract_features(tokens):
    features = []

# implemented bigram
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + "_" + tokens[i+1]
        features.append("bigram=" + bigram)


    return features

# STEP 5: FEATURE MAP
# Our goal is to create mapping: assigning a unique numeric to each feature. Example:
#"word=happy" → 0
#"word=sad"   → 1

def build_feature_map(texts): # builds a dictionary of all features (words) in the training data.
    feature_map = {}
    index = 0

    for text in texts: # process every training sentence
        clean = preprocess(text)
        tokens = tokenize(clean)
        features = extract_features(tokens)

# Adds each feature to dictionary if not already present
        for feat in features:
            if feat not in feature_map:
                feature_map[feat] = index
                index += 1

    return feature_map

# STEP 6: sparse representation (per-sentence form)
def vectorize(text, feature_map):
    features = extract_features(tokenize(preprocess(text)))
    sparse_vector = []
    for feat in features:
        if feat in feature_map:
            sparse_vector.append(feature_map[feat])
    return sparse_vector
# Format: {feature_index: frequency}, for example: [0, 1, 5, 23, 847, 2341]
# means these indexes in the feature map are present in the text, and the rest are not.

# Run the pipeline with training data
#train_texts, train_labels = load_data("isear-train.csv")

#feature_map = build_feature_map(train_texts)
#print(train_texts[0])
#print(preprocess(train_texts[0]))
#print(tokenize(preprocess(train_texts[0])))
#print(extract_features(tokenize(preprocess(train_texts[0]))))
