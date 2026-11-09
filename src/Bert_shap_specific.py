import sys
import csv
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report
import shap
import matplotlib.pyplot as plt

# Label mapping
LABELS = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for i, l in enumerate(LABELS)}


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


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='macro')
    return {'macro_f1': f1}


# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=7,
    id2label=id2label,
    label2id=label2id
)

# Load datasets
train_file = sys.argv[1]
dev_file = sys.argv[2]
test_file = sys.argv[3]

train_texts, train_labels = load_data(train_file)
dev_texts, dev_labels = load_data(dev_file)
test_texts, test_labels = load_data(test_file)

train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
dev_dataset = EmotionDataset(dev_texts, dev_labels, tokenizer)
test_dataset = EmotionDataset(test_texts, test_labels, tokenizer)

# Training arguments — final version
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics
)
trainer.train()
print("Training complete!")

# Dev F1
dev_preds = trainer.predict(dev_dataset)
dev_pred_labels = np.argmax(dev_preds.predictions, axis=-1)
dev_macro_f1 = f1_score(dev_labels, dev_pred_labels, average='macro')
print(f"\n===== DEV SET =====")
print(f"Macro F1: {dev_macro_f1:.4f}")
print(classification_report(dev_labels, dev_pred_labels, target_names=LABELS, digits=3))

# Test F1
test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_macro_f1 = f1_score(test_labels, test_pred_labels, average='macro')
print(f"\n===== TEST SET =====")
print(f"Macro F1: {test_macro_f1:.4f}")
print(classification_report(test_labels, test_pred_labels, target_names=LABELS, digits=3))

# ==================================================================
# SHAP pipeline
# ==================================================================
device = 0 if torch.cuda.is_available() else -1
emotion_pipeline = pipeline(
    "text-classification",
    model=trainer.model,
    tokenizer=tokenizer,
    top_k=None,
    device=device,
    truncation=True,
    max_length=128
)

explainer = shap.Explainer(emotion_pipeline)

# ==================================================================
# 6 target sentences for cross-model comparison
# ==================================================================

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

print("\n===== BERT SPECIFIC EXAMPLE WATERFALL CHARTS =====")

for text, true_label, desc in target_sentences:
    print(f"\n[{desc}]")
    print(f"Text: {text[:100]}...")

    # Compute SHAP values for this single sentence
    shap_values = explainer([text])

    # Get predicted label
    predicted_label = emotion_pipeline(text)[0][0]['label']
    print(f"True: {true_label} | RoBERTa predicted: {predicted_label}")

    # Get SHAP explanation for predicted label
    explanation = shap_values[:, :, predicted_label]
    single_exp = explanation[0]

    # Slice to top 10 by absolute SHAP value to remove "other features"
    order = np.argsort(-np.abs(single_exp.values))[:10]
    top_exp = shap.Explanation(
        values=single_exp.values[order],
        base_values=single_exp.base_values,
        feature_names=np.array(single_exp.feature_names)[order],
    )

    shap.plots.waterfall(top_exp, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(
        f"SHAP waterfall (BERT) — true: '{true_label}', predicted: '{predicted_label}'\n{text[:80]}...",
        fontsize=9
    )
    save_path = f'Bert_shap_{desc}.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")

print("\nAll BERT SHAP waterfall plots saved to the current directory.")
