# Predicting Financial Exclusion in Rwanda

NISR 2026 Big Data Hackathon — Track 2.

This project predicts whether an adult in Rwanda is **financially excluded**, meaning they use no formal or
informal financial services. It uses FinScope Rwanda survey data together with district-level poverty
indicators from EICV. The model is a multilayer perceptron (MLP) with entity embeddings for the categorical
features.

> **Status:** placeholder pipeline. The notebook runs every stage end to end but trains for **one epoch
> only**. Full training, tuning and an XGBoost baseline come later in the ML Pipeline stage.

## Repository contents

| File | Description |
| --- | --- |
| `nisr_2026_financial_exclusion.ipynb` | End-to-end pipeline: setup, data loading, preprocessing, model, training, evaluation, TensorBoard |
| `README.md` | This file |
| `.gitignore` | Files git ignores (checkpoints, caches, virtual environments) |

## Data

| Source | Level | Used for |
| --- | --- | --- |
| [FinScope Rwanda](https://microdata.statistics.gov.rw) | One row per surveyed adult | Demographic features, survey weight, target `financially_excluded` |
| [EICV](https://microdata.statistics.gov.rw) district table | One row per district (30) | `district_poverty_rate`, `district_mean_consumption` |

FinScope and EICV interview different households, so the two tables are joined on **district only**.

Download both files from the NISR microdata portal and place them here:

```
data/finscope_rwanda.csv
data/eicv_district.csv
```

If the real columns have different names, map them to the notebook's names in `COLUMN_MAP` (Section 1).

**If the files are missing, the notebook generates a small synthetic sample with the same schema.** This
proves the pipeline runs, but results from synthetic data mean nothing. Do not report them as findings.

### Features

- **Categorical** (entity embeddings): `district`, `sex`, `education`, `income_source`, `area`
- **Numeric** (standardised): `age`, `household_size`, `owns_phone`, `district_poverty_rate`,
  `district_mean_consumption`
- **Target:** `financially_excluded` (1 = excluded, 0 = included)
- **Weight:** `survey_weight`

## Pipeline

1. **Setup:** imports, fixed random seeds (42) and a single `CONFIG` dictionary holding all settings.
2. **Data loading:** loads FinScope and EICV, joins them on district, and falls back to synthetic data if the
   files are missing.
3. **Preprocessing:**
   - Fills missing values (median for numbers, `"Unknown"` for categories).
   - Splits the data 80/20 into training and validation sets, stratified so the rare excluded class appears in
     both.
   - Learns category vocabularies and scaling statistics from the training set only, to avoid leakage.
   - Weights each training row by its survey weight × a balanced class weight.
4. **Model:**
   - Each categorical feature gets an embedding of size ≤ 10.
   - The embeddings are concatenated with the numeric features.
   - Two dense blocks (128 → 64 units, each Dense → BatchNorm → ReLU → Dropout 0.3) lead to a single sigmoid
     output.
5. **Training:**
   - Adam optimiser (lr 1e-3), binary cross-entropy, batch size 256, 1 epoch.
   - TensorBoard logs scalars, the model graph, weight histograms and HParams.
6. **Evaluation:**
   - Accuracy, AUC, precision and recall on the validation set.
   - A confusion matrix and per-class report at a 0.5 threshold.

## Running it

The notebook is designed for Google Colab but runs in any Jupyter environment with:

```
tensorflow  tensorboard  scikit-learn  pandas  numpy
```

1. Open `nisr_2026_financial_exclusion.ipynb` in Colab or Jupyter.
2. Upload the data files to `data/`, or skip this step to test with synthetic data.
3. Run all cells.
4. TensorBoard opens in the last cell. Check the **Scalars**, **Graphs**, **Histograms** and **HParams** tabs.

Each run writes to its own time-stamped folder under `logs/`.

## Next steps

- Train fully on the real FinScope data and tune the hyperparameters.
- Tune the decision threshold for recall on the excluded class.
- Compare against an XGBoost baseline.

## Team

*Ivan Olivier Muhoza* — *Dorcas Tabitha Akimana*
