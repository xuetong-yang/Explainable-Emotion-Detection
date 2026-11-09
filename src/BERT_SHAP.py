import csv
import numpy as np
import torch
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import shap
import matplotlib.pyplot as plt

# Label mapping
LABELS = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
label2id = {l: i for i, l in enumerate(LABELS)}  # turn label to id, bc the model can only process numbers
id2label = {i: l for i, l in enumerate(LABELS)}  # encode the label bidirectionally, so that in training: label-number, in prediction: number-label


# Load data
def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            if row[0] not in label2id:
                continue
            labels.append(label2id[row[0]])
            texts.append(row[1])
    return texts, labels


# Dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):  # tokenizer the text, and set max_len to 128, which is the default for roberta-base
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)  # truncate if longer than 128, padding if shorter than 128.
        self.labels = labels

    def __len__(self):
        return len(self.labels) 

    def __getitem__(self, idx): 
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}  
        item['labels'] = torch.tensor(self.labels[idx])
    
        return item


# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': f1}


# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=7,  # number of emotion categories
    id2label=id2label,
    label2id=label2id
)

# Load datasets
train_texts, train_labels = load_data('/kaggle/input/datasets/melisakorkmaz/data-set/isear-train.csv')
dev_texts, dev_labels = load_data('/kaggle/input/datasets/melisakorkmaz/data-set/isear-val.csv')
test_texts, test_labels = load_data('/kaggle/input/datasets/melisakorkmaz/data-set/isear-test.csv')

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
dev_dataset = EmotionDataset(dev_texts, dev_labels, tokenizer)

# Training arguments 
training_args = TrainingArguments(
    output_dir='./roberta-isear',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    logging_dir='./logs',
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    seed=42
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model('./roberta-isear-best')
print("Training complete!")

# TEST SET PREDICTIONS
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)
test_output = trainer.predict(test_dataset)
test_pred_ids = np.argmax(test_output.predictions, axis=-1)

test_f1 = f1_score(test_labels, test_pred_ids, average='macro')
print(f"\n===== TEST SET =====")
print(f"Macro F1: {test_f1:.4f}")


# SHAP EXPLAINABILITY
device = 0 if torch.cuda.is_available() else -1
emotion_pipeline = pipeline(
    "text-classification",
    model=trainer.model,
    tokenizer=tokenizer,
    top_k=None,             # return scores for 7 labels
    device=device,
    truncation=True,
    max_length=128
)

# SHAP explainer
explainer = shap.Explainer(emotion_pipeline)


# Stratified random sample, matched to the MLR script:
# n_per_label examples per emotion, shuffled with the same seed

n_per_label = 7  

indices_by_label = defaultdict(list)
for idx, lbl_id in enumerate(test_labels):
    indices_by_label[id2label[lbl_id]].append(idx)

rng = np.random.RandomState(42)
sample_idx = []
for lbl, idxs in indices_by_label.items():
    idxs = idxs.copy()
    rng.shuffle(idxs)
    sample_idx.extend(idxs[:n_per_label])

sample_texts = [test_texts[i] for i in sample_idx]

# SHAP values
# Resulting shape: (n_samples, n_tokens, n_classes)
shap_values = explainer(sample_texts)

MAX_DISPLAY = 10
global_max_abs = 0.4  # fixed axis range, no rescaling


def top_k_explanation(explanation, k=10):
    """Slice an Explanation down to its top-k |SHAP value| features,
    so shap.plots.bar has nothing left to bucket into 'other features'."""
    order = np.argsort(-np.abs(explanation.values))[:k]
    return shap.Explanation(
        values=explanation.values[order],
        base_values=explanation.base_values,
        feature_names=np.array(explanation.feature_names)[order],
    )


emotion_explanations = {}   
for emotion in LABELS:
    agg_explanation = shap_values[:, :, emotion].mean(0)
    emotion_explanations[emotion] = top_k_explanation(agg_explanation, k=MAX_DISPLAY)


def plot_emotion_bar(label, explanation, filename):
    plt.figure()
    shap.plots.bar(explanation, max_display=MAX_DISPLAY, show=False)
    ax = plt.gca()
    ax.set_xlim(-global_max_abs, global_max_abs)
    ax.set_title(f"Top tokens driving '{label}' predictions")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


# Bar plots per emotion labels
for emotion in LABELS:
    filename = f'/kaggle/working/shap_bar_roberta_{emotion}.png'
    plot_emotion_bar(emotion, emotion_explanations[emotion], filename)

print("\nAll 7 SHAP bar plots saved to /kaggle/working/")
