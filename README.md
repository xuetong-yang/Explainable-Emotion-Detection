# Explainable Emotion Detection

Why do emotion classifiers confuse closely related emotions like shame and guilt?
This project compares simple and transformer-based models on 7-class emotion detection, then uses SHAP to explain what drives their predictions.

*TeamLab project, Institute for Natural Language Processing (IMS), University of Stuttgart, 2026.*

---

## Overview

- **Task:** Multi-class emotion classification on the ISEAR dataset (about 6,600 short texts).
- **Labels:** anger, disgust, fear, guilt, joy, sadness, shame.
- **Models:** dummy baselines, Perceptron, Averaged Perceptron, TF-IDF Multinomial Logistic Regression (MLR), fine-tuned RoBERTa.
- **Focus:** explainability. SHAP (SHapley Additive exPlanations) shows which words each model relies on, especially for shame and guilt.

## Results

### Model comparison (test set, 1,145 texts)

| Model | Macro F1 |
|---|---|
| Perceptron | 0.554 |
| Multinomial Logistic Regression (TF-IDF) | 0.560 |
| **RoBERTa (fine-tuned)** | **0.713** |

### Per-class F1 for RoBERTa

| Emotion | Precision | Recall | F1 |
|---|---|---|---|
| joy | 0.892 | 0.914 | **0.902** |
| fear | 0.810 | 0.805 | **0.807** |
| sadness | 0.723 | 0.762 | 0.742 |
| guilt | 0.671 | 0.671 | 0.671 |
| disgust | 0.758 | 0.578 | 0.656 |
| anger | 0.569 | 0.676 | 0.618 |
| shame | 0.602 | 0.591 | 0.597 |

### Key findings

- **A clear gap, not a steady progression.** Both linear models land around 0.56 macro F1. RoBERTa jumps to 0.71.
- **Distinct emotions are easy, related ones are not.** Joy and fear exceed 0.80 F1. Shame, guilt and anger stay below 0.70.
- **Improvement is uneven.** RoBERTa eases some anger-related confusions. But shame and guilt are still confused in both directions, and disgust misclassified as anger even gets worse.
- **SHAP explains why.** Both models rely on the literal emotion words. RoBERTa learns that "ashamed" is a strong two-way signal: it pushes towards shame and away from guilt. The linear model does not.
- **A useful feature is not enough if it is rare.** Only 23% of shame test texts contain such an explicit marker word. So the stronger feature cannot fix the shame/guilt confusion overall.

SHAP plots and heatmaps are in [`src/SHAP_explainability/`](src/SHAP_explainability/).

## Repository structure

```
├── README.md
├── requirements.txt
├── datasets.md                      # Description of the ISEAR dataset
└── src/
    ├── dummy_majority.py            # Majority-class baseline
    ├── dummy_random.py              # Random baseline
    ├── feature_extraction.py        # Unigram features for the Perceptron models
    ├── feature_extraction_bigram.py # Bigram features for the Perceptron models
    ├── perceptron.py                # Perceptron, implemented from scratch
    ├── averaged_perceptron.py       # Averaged Perceptron, implemented from scratch
    ├── MLR_tf_idf_uni-bigram.py     # TF-IDF + Multinomial Logistic Regression (scikit-learn)
    ├── RoBERTa.py                   # RoBERTa fine-tuning (Hugging Face Transformers, PyTorch)
    ├── evaluation.py                # Shared metrics: macro F1, accuracy, per-class F1, confusion matrix
    ├── test_evaluation.py           # Sanity tests for evaluation.py
    ├── MLR_SHAP.py                  # SHAP analysis for MLR (all emotions)
    ├── BERT_SHAP.py                 # SHAP analysis for RoBERTa (all emotions)
    ├── MLR_shap_specific.py         # SHAP waterfall plots for MLR (selected examples)
    ├── Bert_shap_specific.py        # SHAP waterfall plots for RoBERTa (selected examples)
    └── SHAP_explainability/         # SHAP plots and heatmaps
```

## Setup

```bash
git clone https://github.com/xuetong-yang/explainable-emotion-detection.git
cd explainable-emotion-detection
pip install -r requirements.txt
```

The ISEAR dataset is **not included** in this repository because of its license. Please obtain it separately and place the train, validation and test csv files in the project root. See `datasets.md` for details.

## Usage

Run all commands from the project root, not from inside `src/`.

```bash
# Baselines
python src/dummy_majority.py
python src/dummy_random.py

# Perceptron models
python src/perceptron.py isear-train.csv isear-test.csv perceptron_predictions.csv
python src/averaged_perceptron.py isear-train.csv isear-test.csv averaged_perceptron_predictions.csv

# TF-IDF + Multinomial Logistic Regression
python src/feature_extraction.py          # or feature_extraction_bigram.py
python src/MLR_tf_idf_uni-bigram.py

# Fine-tune RoBERTa
python src/RoBERTa.py isear-train.csv isear-val.csv isear-test.csv

# SHAP explanations (all emotions)
python src/MLR_SHAP.py
python src/BERT_SHAP.py

# SHAP waterfall plots for selected examples
python src/MLR_shap_specific.py isear-train.csv isear-val.csv isear-test.csv
python src/Bert_shap_specific.py isear-train.csv isear-val.csv isear-test.csv

# Evaluate any prediction file against the gold labels
python src/evaluation.py isear-test.csv <predictions_file>.csv

# Run the evaluation tests
python src/test_evaluation.py
```

Results are reproducible with fixed random seeds: 42 for MLR and RoBERTa, 123 for the Averaged Perceptron.

## My contribution

This was a two-person project with Melisa Korkmaz.

- Text cleaning and preprocessing of the ISEAR data
- Dummy baselines and the Perceptron and Averaged Perceptron classifiers
- RoBERTa fine-tuning and the shared evaluation module, including tests
- SHAP result analysis and the waterfall scripts for selected examples
- Repository structure, documentation and reproducibility setup

## Authors

- Xuetong Yang
- Melisa Korkmaz
