# AfriSenti — Multilingual Sentiment Analysis

A comprehensive sentiment analysis project for African languages using the AfriSenti Twitter dataset. This project implements state-of-the-art transformer models (XLM-RoBERTa) and LSTM baselines for 3-class sentiment classification (positive, neutral, negative) across 14 African languages.

**Authors:** Ainedembe Denis, Musinguzi Benson  
**Lecturer:** Dr. Sitenda Harriet

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Python Version Compatibility](#python-version-compatibility)

---

## Overview

This project implements a comprehensive analysis of the **AfriSenti dataset**, the largest sentiment analysis dataset for under-represented African languages. The dataset contains **110,000+ annotated tweets** across **14 African languages**, making it an ideal resource for multilingual sentiment analysis research.

### Key Objectives

- Perform sentiment classification on multilingual African language tweets
- Compare transformer-based models (XLM-RoBERTa) with LSTM baselines
- Conduct cross-lingual transfer learning experiments
- Analyze the impact of hyperparameters through ablation studies
- Provide comprehensive evaluation metrics and visualizations

---

## Features

### 1. **Data Exploration & Analysis**
   - Language distribution analysis across 14 languages
   - Text length statistics and visualizations
   - Label distribution analysis (positive, neutral, negative)
   - Dataset sample exploration

### 2. **Advanced Text Preprocessing**
   - **URL removal** and normalization
   - **Emoji handling** (with emoji library support)
   - **User mention** normalization
   - **Hashtag** processing
   - **Punctuation** normalization
   - **Whitespace** cleaning

### 3. **Model Implementations**
   - **XLM-RoBERTa** (Transformer-based, multilingual)
   - **LSTM Baseline** (Bidirectional LSTM with embeddings)
   - Fine-tuning with early stopping and gradient clipping
   - Learning rate scheduling for transformers

### 4. **Comprehensive Evaluation**
   - **Accuracy** and **F1-Score** (macro-averaged)
   - **ROC-AUC** (multi-class, macro-averaged)
   - **Confusion matrices** with visualizations
   - **Example predictions** with probability distributions
   - **Attention visualization** for transformer models

### 5. **Ablation Studies**
   - Hyperparameter sensitivity analysis
   - Batch size variations (8, 16, 32)
   - Learning rate variations (1e-5, 2e-5, 5e-5)
   - Sequence length variations (64, 128, 256)
   - Performance visualizations and heatmaps

### 6. **Cross-Lingual Transfer Learning**
   - Train on one language, test on another
   - Multiple language pair experiments:
     - Swahili → Amharic
     - Swahili → Pidgin English (pcm)
   - Cross-lingual performance analysis

---

## Dataset

### AfriSenti Twitter Dataset

The **AfriSenti-Twitter** dataset from HausaNLP contains sentiment-annotated tweets in 14 African languages

**Dataset Source:** [HausaNLP/AfriSenti-Twitter](https://huggingface.co/datasets/HausaNLP/AfriSenti-Twitter) on Hugging Face

---

## Methodology

### 1. **Data Preprocessing**
   - Text cleaning (URLs, mentions, hashtags)
   - Emoji normalization
   - Multilingual tokenization using XLM-RoBERTa tokenizer
   - Label encoding (string → integer)

### 2. **Model Training**
   - **XLM-RoBERTa**: Fine-tuned for 3-5 epochs with:
     - Early stopping (patience=2)
     - Gradient clipping (max_norm=1.0)
     - Learning rate scheduling with warmup
     - Batch size: 8-32 (varies in ablation studies)
   
   - **LSTM Baseline**: Trained from scratch with:
     - Bidirectional LSTM layers
     - Embedding layer
     - Early stopping (patience=2)
     - Gradient clipping

### 3. **Evaluation Metrics**
   - **Accuracy**: Overall classification accuracy
   - **F1-Score**: Macro-averaged F1-score across all classes
   - **ROC-AUC**: Multi-class ROC-AUC (one-vs-rest, macro-averaged)
   - **Confusion Matrix**: Per-class performance visualization

### 4. **Experiments**
   - **Baseline Comparison**: XLM-RoBERTa vs LSTM
   - **Ablation Studies**: Hyperparameter sensitivity analysis
   - **Cross-Lingual Transfer**: Language transfer experiments

---

## Project Structure

```
Sentiment-Analysis-Multilingual-Tweets/
├── afrisenti-analysis.ipynb    # Main Jupyter notebook with complete analysis
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation (this file)
├── best_lstm.pt                # Saved LSTM model checkpoint
└── outputs/                    # Generated visualizations and outputs
```

---

## Installation

### Prerequisites

- Python 3.11 or earlier (see [Python Version Compatibility](#python-version-compatibility))
- pip or conda package manager
- Jupyter Notebook or JupyterLab

### Step 1: Clone or Download the Repository

```bash
git clone <repository-url>
cd Sentiment-Analysis-Multilingual-Tweets
```

### Step 2: Create Virtual Environment (Recommended)

**Using conda:**
```bash
conda create -n afrisenti python=3.11
conda activate afrisenti
```

**Using venv:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install Additional NLTK Data

Open a Python terminal or Jupyter notebook and run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### Step 5: (Optional) Install spaCy Language Models

For English:
```bash
python -m spacy download en_core_web_sm
```

For multilingual support:
```bash
python -m spacy download xx_ent_wiki_sm
```

---

## Usage

### Running the Notebook

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Open `afrisenti-analysis.ipynb` in your browser

3. **Run cells sequentially:**
   - The notebook is organized into sections:
     1. Imports & Setup
     2. Dataset Loading
     3. Data Exploration
     4. Text Preprocessing
     5. Model Implementation
     6. Training
     7. Evaluation
     8. Ablation Studies
     9. Cross-Lingual Testing

### Quick Start Example

```python
# Load dataset
from datasets import load_dataset
dataset = load_dataset("HausaNLP/AfriSenti-Twitter", "amh", trust_remote_code=True)

# View sample
print(dataset['train'][0])
# Output: {'tweet': '...', 'label': 2}
```

### Training a Model

The notebook includes complete training pipelines for both models:

```python
# Train LSTM baseline
train_lstm(lstm_model, train_loader, val_loader, epochs=3, lr=1e-3, patience=2)

# Train XLM-RoBERTa transformer
train_transformer(transformer_model, train_loader, val_loader, 
                  epochs=5, lr=2e-5, patience=2)
```

---

## Requirements

### Python Version Compatibility

**Important:** This project requires **Python 3.11 or earlier** for transformer models to work properly.

**If using Python 3.13:**
- LSTM baseline model will work fine
- Transformer models (XLM-RoBERTa, mBERT) will **not load** due to compatibility issues

### Core Dependencies

#### Deep Learning
- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.30.0` - Hugging Face transformers
- `tensorflow>=2.13.0` - TensorFlow (optional, for some utilities)

#### Data Processing
- `datasets<4.0.0` - Hugging Face datasets (version constraint for AfriSenti compatibility)
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing

#### NLP & Text Processing
- `nltk>=3.8.0` - Natural language toolkit
- `spacy>=3.5.0` - Advanced NLP library
- `gensim>=4.3.0` - Word embeddings
- `sentence-transformers>=2.2.0` - Sentence embeddings
- `emoji>=2.8.0` - Emoji handling

#### Visualization
- `matplotlib>=3.7.0` - Plotting library
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.14.0` - Interactive visualizations
- `scikit-learn>=1.3.0` - Machine learning utilities

#### Utilities
- `tqdm>=4.65.0` - Progress bars
- `ipywidgets>=8.0.0` - Jupyter widgets

### Optional Dependencies

- `librosa>=0.10.0` - Audio processing (if needed)
- `torchaudio>=2.0.0` - Audio for PyTorch (if needed)
- `speechbrain>=0.5.0` - Speech processing (if needed)

---

## Python Version Setup

### Using Conda (Recommended)

```bash
# Create environment with Python 3.11
conda create -n afrisenti python=3.11

# Activate environment
conda activate afrisenti

# Install dependencies
pip install -r requirements.txt
```

### Using venv

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Setting Up Jupyter Kernel

If using a virtual environment, register it as a Jupyter kernel:

```bash
pip install ipykernel
python -m ipykernel install --user --name afrisenti --display-name "Python 3.11 (AfriSenti)"
```

Then select the kernel in Jupyter Notebook.

---

## Notebook Sections

The main notebook (`afrisenti-analysis.ipynb`) is organized into the following sections:

1. **Imports & Setup** - Library imports and environment setup
2. **Dataset Loading** - Loading AfriSenti dataset from Hugging Face
3. **Data Exploration** - Language distribution, text length, label analysis
4. **Text Preprocessing** - URL removal, emoji handling, normalization
5. **Model Implementation** - XLM-RoBERTa and LSTM model definitions
6. **Training** - Training functions with early stopping and gradient clipping
7. **Evaluation** - Comprehensive metrics, predictions, attention visualization
8. **Ablation Studies** - Hyperparameter sensitivity analysis
9. **Cross-Lingual Testing** - Language transfer experiments

---

## Contributing

This is an academic project. For questions or suggestions, please contact the authors.

---

## License

This project is for academic and research purposes.

---

## Acknowledgments

- **Dataset**: AfriSenti-Twitter dataset by HausaNLP
- **Models**: XLM-RoBERTa by Facebook AI Research
- **Framework**: Hugging Face Transformers
- **Supervision**: Dr. Sitenda Harriet

---

## Contact

For questions or issues, please contact:
- **Ainedembe Denis** - dembedenis@gmail.com, ainedembe.denis@stud.umu.ac.ug
- **Musinguzi Benson** - musiben@gmail.com, musinguzi.benson@stud.umu.ac.ug

---

**Last Updated:** Dec 2025
