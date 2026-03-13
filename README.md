# News Source Classification: Fox News vs NBC News

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers/)

A deep learning approach to binary news source classification using transformer-based models. This project implements and compares RNN-based architectures (Simple RNN, GRU, LSTM) with pre-trained transformer models (DistilBERT) for classifying news headlines as either **Fox News** or **NBC News**.

**Course**: CIS 4190/5190 Applied Machine Learning, University of Pennsylvania, Fall 2025

## Results Summary

| Model | Validation Accuracy | Leaderboard Accuracy | Inference Time |
|-------|---------------------|----------------------|----------------|
| Logistic Regression (baseline) | 63.0% | - | - |
| Simple RNN | 86.0% | - | - |
| Bidirectional GRU | 87.0% | - | - |
| Bidirectional LSTM | 88.0% | - | - |
| **DistilBERT (Ours)** | **98.0%** | **93.26%** | 10.45 ms |

Our DistilBERT model achieved **top-tier performance** on the course leaderboard, demonstrating robust generalization across multiple evaluation datasets.

## Project Overview

### Problem Statement
Given a news article URL, classify whether it originates from Fox News or NBC News based solely on the headline text extracted from the URL slug. This task captures meaningful differences in writing style, phrasing, and semantic emphasis across news outlets.

### Key Contributions
1. **Comprehensive Model Comparison**: Systematic evaluation of RNN variants (Simple RNN, GRU, LSTM) against transformer-based architectures
2. **Data Leakage Investigation**: Identified and addressed potential data leakage from NBC-specific URL patterns (e.g., `rcna` suffix)
3. **Transfer Learning Analysis**: Demonstrated the effectiveness of fine-tuning pre-trained language models for domain-specific text classification
4. **Efficient Model Selection**: Achieved state-of-the-art results with DistilBERT, balancing accuracy and inference efficiency

## Repository Structure
```
├── model.py              # DistilBERT classifier architecture
├── preprocess.py         # URL parsing and text extraction pipeline
├── model.pt              # Trained model weights
├── CIS_519_Final_Project.ipynb  # Training notebook with experiments
└── README.md
```

## Model Architecture

### DistilBERT Classifier
```
Input Text → DistilBERT Tokenizer → DistilBERT Encoder → [CLS] Token → Dropout(0.1) → Linear(768, 2) → Output
```

**Architecture Details**:
- **Encoder**: DistilBERT-base-uncased (6 transformer layers, 768 hidden dimensions, 12 attention heads)
- **Classification Head**: Single linear layer mapping 768-dimensional [CLS] representation to 2 output classes
- **Regularization**: Dropout (p=0.1) before the classification layer

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| Weight Decay | 0.01 |
| Batch Size | 32 |
| Epochs | 3 |
| Max Sequence Length | 128 |
| LR Schedule | Linear warmup (10%) + linear decay |
| Gradient Clipping | Max norm 1.0 |

## Dataset

- **Size**: 8,500 URLs (balanced: 4,250 Fox News, 4,250 NBC News)
- **Source**: Public XML sitemaps from foxnews.com and nbcnews.com
- **Split**: 85% training / 15% validation (stratified)
- **Preprocessing**: URL slug extraction with deterministic parsing rules

**Dataset available on Hugging Face**: [8k5 News URL Dataset](https://huggingface.co/datasets/xinjiehu76/cis5190-25f-projectb-8k5-combined)

### Data Preprocessing Pipeline
```python
URL: https://www.foxnews.com/politics/senate-passes-major-legislation
                    ↓
Extract Path: /politics/senate-passes-major-legislation
                    ↓
Parse Slug: senate-passes-major-legislation
                    ↓
Clean Text: "senate passes major legislation"
```

**Data Leakage Mitigation**: NBC News URLs contain article ID suffixes (e.g., `rcna240477`). We remove these identifiers to ensure the model learns from semantic content rather than URL patterns.

## Quick Start

### Installation
```bash
pip install torch transformers pandas scikit-learn
```

### Inference
```python
from model import Model
import torch

# Load model
model = Model()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Predict
headlines = ["senate passes major legislation", "breaking news update"]
predictions = model.predict(headlines)
print(predictions)  # ['foxnews', 'nbcnews']
```

### Training
```bash
# Open the Jupyter notebook
jupyter notebook CIS_519_Final_Project.ipynb
```

Or use Google Colab with GPU runtime for faster training.

## Experimental Results

### RNN vs Transformer Comparison

Our experiments demonstrate that transformer-based models significantly outperform RNN variants for this task:

- **DistilBERT vs LSTM**: +10 percentage points improvement
- **Key Advantages**:
  1. Transfer learning from large-scale pre-training
  2. Self-attention captures global context effectively
  3. WordPiece tokenization handles OOV words gracefully

### Data Leakage Analysis

| Preprocessing | Validation Acc | Leaderboard Acc |
|---------------|----------------|-----------------|
| Keep ID suffix | 98.16% | 93.34% |
| Remove ID suffix | 97.80% | 93.26% |

The minimal difference (<1%) confirms that our model learns genuine stylistic differences rather than exploiting URL patterns.

### Optimizer Comparison (LSTM)

| Optimizer | lr=2e-3 | lr=1e-3 | lr=5e-4 | lr=3e-4 |
|-----------|---------|---------|---------|---------|
| Adam | 87.06% | 88.31% | 85.96% | 86.12% |
| AdamW | 88.39% | 87.84% | 86.35% | 85.73% |

## Technical Details

### Why DistilBERT?

We evaluated BERT-base (110M parameters) vs DistilBERT (66M parameters):

| Model | Parameters | Accuracy | Inference Speed |
|-------|------------|----------|-----------------|
| BERT-base | 110M | ~98% | Baseline |
| DistilBERT | 66M | ~98% | 40% faster |

DistilBERT retains 97% of BERT's language understanding while being significantly more efficient, making it ideal for this binary classification task.

### Transfer Learning Strategy

Since news headline classification is closely aligned with DistilBERT's pre-training objective (English text understanding), we:
- Fine-tune **all layers** with a uniform learning rate
- Use a **single linear layer** as the classification head (no deep MLP needed)
- Apply **linear warmup** followed by decay for stable optimization

## References

- Sanh, V., et al. (2019). [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). arXiv preprint arXiv:1910.01108.
- Devlin, J., et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). arXiv preprint arXiv:1810.04805.
