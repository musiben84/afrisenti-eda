# AfriSenti Multilingual Sentiment — Short Report Template
Date: 2025-11-25

## 1. Dataset Overview & Preprocessing
- Languages included: Swahili (sw), Amharic (am), English (en).
- Cleaning: URLs→`URL`, mentions→`@USER`, hashtags→`HASHTAG_*`, emojis demojized, simple slang normalization.
- Tokenization: `xlm-roberta-base` (switchable to AfriBERTa). Max length = 128.

## 2. EDA Highlights
- Language distribution: see `outputs/figures/lang_distribution.png`.
- Text lengths (char/token) and per‑language boxplots.
- Label imbalance overall and by language.

## 3. Models
- Transformer: XLM‑R/AfriBERTa fine‑tuned for 3‑class sentiment (3–5 epochs, early stopping, grad clipping).
- Baseline: LSTM using tokenizer’s subword indices.

## 4. Results
- Metrics: Accuracy, Macro‑F1, ROC‑AUC(OVR), Confusion Matrix.
- Example predictions and attention heatmap.
- Ablation results (batch size, LR, max length) [optional].
- Cross‑lingual results (train on sw → test on en/am) [optional].

## 5. Reflection
- What improved performance (e.g., normalization, model choice, fine‑tuning steps).
- What to try next (longer training, better slang dictionaries, per‑language fine‑tuning, class rebalancing).
