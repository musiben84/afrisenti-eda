# AfriSenti — Multilingual Sentiment Analysis

Sentiment Analysis (Multilingual Tweets) 

**Authors:** Ainedembe Denis, Musinguzi Benson  
**Lecturer:** Dr. Sitenda Harriet

This project implements a comprehensive analysis of the AfriSenti dataset, the largest sentiment analysis dataset for under-represented African languages, covering 110,000+ annotated tweets in 14 African languages.

## Requirements

### Python Version Compatibility

**Important:** This project requires Python 3.11 or earlier for transformer models to work properly.

- **Python 3.13:** Has compatibility issues with transformers library (models cannot load)
- **Python 3.11/3.12:** Recommended - all features work correctly
- **Python 3.10:** Also compatible

**If you're using Python 3.13:**
- You can still use the LSTM baseline model (works fine)
- Transformer models (XLM-RoBERTa, mBERT) will not load

### Installation

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

Or install packages individually in a Jupyter notebook:

```python
%pip install -q "datasets<4.0.0" pandas numpy
%pip install -q torch tensorflow transformers
%pip install -q librosa torchaudio speechbrain
%pip install -q nltk spacy gensim sentence-transformers
%pip install -q matplotlib seaborn plotly scikit-learn
%pip install -q ipywidgets tqdm
```

### Required Libraries

#### Deep Learning
- **PyTorch** (`torch`) - Deep learning framework
- **TensorFlow/Keras** (`tensorflow`) - Deep learning framework
- **Hugging Face Transformers** (`transformers`) - Pre-trained transformer models

#### Speech Processing
- **librosa** - Audio and music analysis
- **torchaudio** - Audio processing for PyTorch
- **speechbrain** - Speech processing toolkit

#### Text Processing
- **NLTK** (`nltk`) - Natural language toolkit
- **spaCy** (`spacy`) - Advanced NLP library
- **Gensim** (`gensim`) - Topic modeling and word embeddings
- **datasets** (`datasets<4.0.0`) - Hugging Face datasets library (version constraint for AfriSenti compatibility)
- **sentence-transformers** - Sentence embeddings

#### Visualization & Analysis
- **matplotlib** - Plotting library
- **seaborn** - Statistical data visualization
- **plotly** - Interactive visualizations
- **scikit-learn** - Machine learning library (includes t-SNE and PCA)

#### Data Processing
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

#### Utilities
- **tqdm** - Progress bars
- **ipywidgets** - Interactive widgets for Jupyter

### Additional Setup

#### NLTK Data
NLTK requires downloading additional data. Run in Python/Jupyter:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

#### spaCy Language Models
spaCy requires downloading language models separately. For English:

```bash
python -m spacy download en_core_web_sm
```

For multilingual support:

```bash
python -m spacy download xx_ent_wiki_sm
```

## Python Version Setup (For Transformer Models)

If you need to use transformer models (XLM-RoBERTa, mBERT), Python v3.11 is neeed.
- Installing Python 3.11 alongside Python 3.13
- Creating a virtual environment
- Setting up Jupyter kernel

**Quick Start (Conda):**
```bash
conda create -n afrisenti python=3.11
conda activate afrisenti
pip install -r requirements.txt
```

## Folder Layout
```
├── afrisenti-analysis.ipynb    # Main analysis notebook
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── outputs/                    # Generated outputs
```

