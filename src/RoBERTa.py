import csv
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

train_file = sys.argv[1]
dev_file = sys.argv[2]
test_file = sys.argv[3]

# Label mapping
LABELS = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
label2id = {l: i for i, l in enumerate(LABELS)} # turn label to id, bc the model can only process numbers
id2label = {i: l for i, l in enumerate(LABELS)} # encode the label bidirectionally, so that in training: label-number, in prediction: number-label

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
    def __init__(self, texts, labels, tokenizer, max_len=128): # tokenizer the text, and set max_len to 128, which is the default for roberta-base
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_len) # truncate if longer than 128, padding if shorter than 128.
        self.labels = labels

    def __len__(self):
        return len(self.labels) # return the number of samples in the dataset

    def __getitem__(self, idx): # return the item at index idx
        # encoding is a dictionary with keys 'input_ids', 'attention_mask', and 'token_type_ids'
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()} # get value at index idx for each key in the encoding, and convert it to a tensor
        item['labels'] = torch.tensor(self.labels[idx])
        # add the right label of the sample to the item, so that the trainer can calculate the loss during training
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
    num_labels=7,
    id2label=id2label,
    label2id=label2id
)

# Load datasets

train_texts, train_labels = load_data(train_file)
dev_texts, dev_labels = load_data(dev_file)
test_texts, test_labels = load_data(test_file)

# Training arguments
training_args = TrainingArguments(
    output_dir='./roberta-isear',
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True, 
    logging_dir='./logs',
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio =0.1,
    seed=42 
    # we also run this code on seed 123 and 456 to testify if the results are consistent, and they are. 
    # So we use the results of seed 42 as the final results.
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

# Per-class F1 for dev
dev_preds = trainer.predict(dev_dataset) # get the predictions for the dev dataset
dev_pred_labels = np.argmax(dev_preds.predictions, axis=-1) # get the predicted labels by taking the argmax of the predictions
dev_macro_f1 = f1_score(dev_labels, dev_pred_labels, average='macro') # calculate the macro F1 score for the dev dataset
print("Dev Macro F1:", dev_macro_f1)
print("\n===== PER-CLASS F1 (DEV) =====")
print(classification_report(dev_labels, dev_pred_labels, target_names=LABELS, digits=3)) 

# Per-class F1 for test
test_preds = trainer.predict(test_dataset)
test_pred_labels = np.argmax(test_preds.predictions, axis=-1)
test_macro_f1 = f1_score(test_labels, test_pred_labels, average='macro')
print("Test Macro F1:", test_macro_f1)
print("\n===== PER-CLASS F1 (TEST) =====")
print(classification_report(test_labels, test_pred_labels, target_names=LABELS, digits=3))

# Confusion matrix
cm_dev = confusion_matrix(dev_labels, dev_pred_labels, labels=list(range(7)))
# input true labels and predicted labels
# labels=list(range(7)) make sure all 7 classes are ordered 0-6, even if some classes are not present in the dev dataset, so that the confusion matrix is always 7x7
disp_dev = ConfusionMatrixDisplay(confusion_matrix=cm_dev, display_labels=LABELS)
# transfer the number matrix to a plot with labels. turn 0-6 to the corresponding emotion labels.
disp_dev.plot()
plt.tight_layout()
plt.savefig('confusion_matrix_roberta_dev.png', dpi=150, bbox_inches='tight')
plt.close()

cm_test = confusion_matrix(test_labels, test_pred_labels, labels=list(range(7)))
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=LABELS) 
disp_test.plot()
plt.tight_layout()
plt.savefig('confusion_matrix_roberta_test.png', dpi=150, bbox_inches='tight')
plt.close()

import pandas as pd
df = pd.DataFrame({
    'text': test_texts,
    'true_label': [id2label[l] for l in test_labels],
    'roberta_pred': [id2label[l] for l in test_pred_labels]
})
df.to_csv('roberta_predictions.csv', index=False)
