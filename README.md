# AI Homework 3 — NLP

**AI-Generated Text Detection | Ukrainian Catholic University, 2026**

This project explores multilingual AI-generated text detection using traditional machine learning, transformer-based models, and stacking ensembles.

The task is to classify English and German texts into three categories: `human`, `ai`, and `fusion` (human-written text with AI-generated or rewritten parts).

## Dataset

- **Training:** 272,352 labeled texts
- **Test:** 25,000 unlabeled texts
- **Languages:** English and German
- **Domains:** 8
- **Evaluation metric:** Macro-F1

A group-aware validation strategy was used to reduce data leakage between related documents.

## Models and results

| Approach | Validation Macro-F1 |
|---|---:|
| TF-IDF + Logistic Regression | 0.87852 |
| XLM-RoBERTa (head-tail) | 0.87345 |
| TF-IDF + XLM-R ensemble | 0.91596 |
| **Final stacking ensemble** | **0.96556 (OOF)** |

**Best public Kaggle score: 0.96047**

The final approach combines TF-IDF, multiple XLM-RoBERTa models, multilingual DeBERTa, long-document window predictions, and a stacking ensemble of Histogram Gradient Boosting and Logistic Regression.

## Repository structure

- `hw3_eda.ipynb` — Exploratory data analysis and validation strategy
- `hw3_baseline.ipynb` — TF-IDF baseline and alternative approaches
- `hw3_XLM_RoBERTa.ipynb` — Transformer fine-tuning and head-tail inference
- `hw3_my_approach.ipynb` — Ensemble experiments and feature engineering
- `hw3_explainability.ipynb` — Feature importance, LIME, and error analysis
- `hw3_final_pipeline_v3.ipynb` — Final stacking pipeline and submission generation

## Reproducibility

The experiments were developed in Google Colab using Python, scikit-learn, PyTorch, and Hugging Face Transformers.

To reproduce the final predictions:

1. Obtain the competition dataset from Kaggle.
2. Configure Kaggle credentials and Google Drive access in Colab.
3. Run the notebooks in the order listed above.
4. Provide the saved transformer checkpoints and prediction caches required by the final pipeline.
5. Run `hw3_final_pipeline_v3.ipynb` to generate the submission CSV.

The saved checkpoints and caches are required for reproduction; the notebooks alone do not contain the trained model weights.

## Author

Daryna Shevchuk  
BSc Computer Science — Ukrainian Catholic University
