import csv, sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# updated for testing of scoring functions without running the main script
# Scoring logic is split into functions (compute_accuracy, compute_macro_f1)
# so test_evaluation.py can import them without running the rest of the script. 

def compute_accuracy(gold_labels, pred_labels):
    match = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p) # count how many predictions match the true label
    return match / len(gold_labels)


def compute_macro_f1(gold_labels, pred_labels, labels):
    f1_scores = []
    for label in labels:
        # TP: correct predictions for this label
        TP = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p == label)
        # FP: wroong predictions for this label
        FP = sum(1 for g, p in zip(gold_labels, pred_labels) if g != label and p == label)
        # FN: this was the true label, but model predicted something else
        FN = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label and p != label)
        # both are edge cases, so we guard against dividing by zero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores) # macro average: every label counts equally, no matter how many examples it has


def main():
    goldfile = sys.argv[1]
    dummyfile = sys.argv[2]

    with open(goldfile, 'r') as gold:
        reader = csv.reader(gold)
        goldrows = list(reader)

    with open(dummyfile, 'r') as dummy:
        reader = csv.reader(dummy)
        dummyrows = list(reader)

    valid_labels = {'joy', 'fear', 'shame', 'disgust', 'guilt', 'anger', 'sadness'}

    goldlabels = []
    dummylabels = []
    texts = []

    for gold_row, dummy_row in zip(goldrows, dummyrows):     # the gold file and the predictions file are aligned row-by-row, so we walk through both files together
        if len(gold_row) < 2 or len(dummy_row) < 2: # # skip anything that isn't a proper row
            continue
        if gold_row[0] not in valid_labels: # # skip rows where the true label isn't one of the 7 emotions
            continue
        goldlabels.append(gold_row[0])
        dummylabels.append(dummy_row[0])
        texts.append(gold_row[1])
        
    unique_labels = []  # build the list of labels actually seen in the gold data, keeping the order they first appear in
    for label in goldlabels:
        if label not in unique_labels:
            unique_labels.append(label)

    accuracy = compute_accuracy(goldlabels, dummylabels)
    print("accuracy:", accuracy)

    marcoF1 = compute_macro_f1(goldlabels, dummylabels, unique_labels)
    print("marcoF1:", marcoF1)

    # error case analysis
    # print("\n--- Error Cases: shame predicted as anger ---")
    # for i in range(len(goldlabels)):
    #     if goldlabels[i] == 'shame' and dummylabels[i] == 'anger':
    #         print(texts[i])
    #         print()

    # print("\n--- Error Cases: guilt predicted as anger ---")
    # for i in range(len(goldlabels)):
    #     if goldlabels[i] == 'guilt' and dummylabels[i] == 'anger':
    #         print(texts[i])
    #         print()

    # confusion matrix
    # labels = ['anger', 'disgust', 'fear', 'guilt', 'joy', 'sadness', 'shame']
    # cm = confusion_matrix(goldlabels, dummylabels, labels=labels)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    # disp.plot()
    # plt.savefig('confusion_matrix.png')


if __name__ == '__main__':
    main()
