# AfriSenti — Multilingual Sentiment Analysis (VS Code + Jupyter)

GROUP -2 Assignment
Sentiment Analysis (Multilingual Tweets) 

Aurthors: Ainedembe Denis 
          Musinguzi Benson

Lecturer: Dr. Sitenda Harriet


This project walks you from **initial EDA** through **multilingual modeling** (XLM‑R/AfriBERTa + LSTM baseline) for 3‑class sentiment (positive/neutral/negative). It covers the assignment items end‑to‑end.

## Folder Layout
```
afrisenti-eda/
  environment.yml
  README.md
  data/                      # put CSVs here (or use HuggingFace loader in notebooks)
  notebooks/
    01_AfriSenti_EDA.ipynb
    02_AfriSenti_Modeling.ipynb
  outputs/
    figures/
    tables/
  reports/
    Report_Template.md
```

## 0) Open in VS Code and create the environment
1. Open this folder in **VS Code**.
2. In the built‑in Terminal (PowerShell on Windows), run:
   ```powershell
   conda env create -f environment.yml
   conda activate afrisenti
   python -m ipykernel install --user --name afrisenti --display-name "Python (afrisenti)"
   ```

## 1) Put the data
- If you have AfriSenti CSVs, copy them into `data/`. The notebooks auto‑detect columns: `text`/`tweet`, `label`/`sentiment`, `lang`/`language`.
- Or load directly from **HuggingFace** by uncommenting and editing the dataset IDs in the notebooks.

---
# Notebook 1 — 01_AfriSenti_EDA.ipynb (Items 1–2 from your brief)
**Covers:**
- Load AfriSenti (Swahili, Amharic, English)
- Missing values & duplicates; de‑duplication
- **Language distribution** (bar plot) — (Task 2a)
- **Text length** (characters/tokens, per‑language) — (Task 2b)
- **Label imbalance** overall & per language — (Task 2c)
- Saves figures to `outputs/figures/` and tables/summary to `outputs/tables/`

Run:
```powershell
jupyter lab
# open notebooks/01_AfriSenti_EDA.ipynb → select kernel "Python (afrisenti)" → Run All
```

---
# Notebook 2 — 02_AfriSenti_Modeling.ipynb (Items 19–24)
**Covers:**
- 19. Preprocessing with multilingual tokenizers (mBERT/XLM‑R), handling **emojis**, **URLs**, **mentions**, **hashtags**, and a simple **slang** normalizer.
- 20. Fine‑tune **XLM‑RoBERTa** (default) or switch to **AfriBERTa**; compare with an **LSTM baseline**.
- 21. Train **3–5 epochs** with **early stopping** and **gradient clipping**.
- 22. Evaluate **accuracy**, **macro‑F1**, **ROC‑AUC (OVR)**, **confusion matrix**, print **example predictions**, and show an **attention heatmap**.
- 23. **Ablation** helpers to vary **batch size**, **learning rate**, **sequence length**; results saved to CSV.
- 24. **Cross‑lingual testing** helpers (e.g., train on Swahili, test on English/Amharic).

Outputs saved under `outputs/`:
- `outputs/figures/afrisenti_confusion_transformer.png`
- `outputs/figures/attention_heatmap_sample.png`
- `outputs/tables/ablations_results.csv` (if you run ablations)
- `outputs/transformers_runs/` (HF Trainer checkpoints)
- `outputs/lstm_best.pt` (LSTM weights)

---
## Tips
- If Amharic glyphs do not render, install a font with Ethiopic coverage (e.g., *Noto Sans Ethiopic*) and set it in the first cell:
  ```python
  import matplotlib.pyplot as plt
  plt.rcParams["font.family"] = "Noto Sans Ethiopic"
  ```
- For numeric labels, enable the mapping to text labels (`negative/neutral/positive`) in the notebooks.
- If runtime is long, first run on a small subset, then scale up.
