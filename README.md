# Explainable Signed Link Prediction for Reddit Inter-Community Conflict Warning

## Project Overview

This repository contains a paper-style final project for the Social Media Data Analysis course. The project studies how Reddit communities link to each other and predicts whether a future source-target subreddit relationship becomes negative-dominant using temporal signed-network features.

Recommended project title:

> Predicting Negative Cross-Community Hyperlinks on Reddit Using Temporal Signed Network Features

The course-project scope is **negative hyperlink prediction**. The longer-term research extension is **explainable early warning of inter-community conflict**.

## Research Questions

1. Which subreddits are major sources and targets of negative cross-community hyperlinks?
2. Do signed network features improve prediction compared with text-only features?
3. Does a hybrid model that combines text, graph, and temporal history outperform simple baselines?
4. Which historical features explain future negative-dominant inter-community relationships?

## Dataset

The project uses the Reddit Hyperlink Network from the Kaggle `Signed Graphs` mirror, with Stanford SNAP/Kumar et al. cited as the original academic source:

- `soc-redditHyperlinks-body.tsv`
- `soc-redditHyperlinks-title.tsv`

Course-access source: <https://www.kaggle.com/datasets/wolfram77/graphs-signed>

Original source: <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>

The raw files should be placed in `data/raw/`. They are not committed because of size.

Expected raw schema:

- `SOURCE_SUBREDDIT`
- `TARGET_SUBREDDIT`
- `POST_ID`
- `TIMESTAMP`
- `LINK_SENTIMENT`
- `PROPERTIES`

The `PROPERTIES` column contains 86 numeric text-property features. The pipeline parses these into `text_property_00` to `text_property_85` for text-only and hybrid ablation experiments.

The reproducible dataset audit is documented in `docs/data_provenance.md` and can be rerun with:

```bash
python scripts/audit_dataset.py --raw-dir data/raw --json-out data/processed/dataset_audit.json
```

## Methodology

The implemented workflow has four phases:

1. **Data preparation**
   - Load body/title TSV files.
   - Standardize column names and timestamps.
   - Concatenate both files and add `dataset_source`.
   - Apply optional k-core filtering for a denser modeling graph.

2. **Network construction and feature engineering**
   - Build a directed signed multigraph.
   - Extract node features: in/out degree, signed degree, PageRank, betweenness, reciprocity, clustering coefficient, and community-level negative ratios.
   - Extract pair features: interaction count, positive/negative counts, negative ratio, reciprocal edge.
   - Extract community-pair features: same-community flag, community size ratio, and community negativity gap.
   - Extract structural-balance features from signed local neighborhoods.
   - Aggregate 86 text-property features at pair level.

3. **Strict temporal modeling**
   - Train features are computed only from interactions before the history cutoff.
   - Labels are computed from a disjoint future window.
   - A pair is labeled negative when future negative hyperlinks outnumber future positive/neutral hyperlinks in that window.
   - Models are compared using graph-only, text-only, and hybrid feature sets.
   - Ablations include no-balance graph/hybrid settings and a history-only feature set.
   - Baselines include dummy prior and historical negative-ratio heuristics.
   - Decision thresholds are tuned on validation data and applied once to the test set.

4. **Report figures and interpretation**
   - Label distribution.
   - Monthly negative-link ratio.
   - Top negative source and target subreddits.
   - Degree distribution.
   - Community-level negative ratio.
   - Readable subreddit network sample colored by detected community.
   - Community-pair negative-ratio heatmap.
   - Model comparison by PR-AUC.
   - Precision-recall and ROC curves.
   - Confusion matrix.
   - Feature importance.
   - Threshold trade-off.
   - Sampled k-core robustness probe.
   - Representative true-positive, false-positive, and false-negative cases.

## Evaluation Metrics

Because negative links are a minority class, accuracy is not the main metric. The report should emphasize:

- PR-AUC
- ROC-AUC
- F1 for the negative class
- Macro-F1
- Precision and recall
- Balanced accuracy
- Confusion matrix

## Latest Verified Result

The latest verified run is summarized in `docs/run_summary.md`. The exported metrics include 41 model/feature-set rows across Logistic Regression, Random Forest, XGBoost, LightGBM, dummy baselines, and a historical negative-ratio heuristic.

Best test result by PR-AUC:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | Logistic Regression | 0.1840 | 0.7569 | 0.2700 | 0.2050 | 0.3954 |

Additional report-hardening artifacts:

- `data/processed/phase3/robustness_metrics.csv`
- `data/processed/phase3/error_analysis_cases.csv`
- `reports/figures/robustness_kcore_pr_auc.png`
- `reports/figures/threshold_tradeoff.png`
- `docs/final_presentation.pptx`

## Installation

Recommended environment: Python 3.11 or 3.12. The original local run used a Python 3.14 `.venv`, but Python 3.11/3.12 is more portable for ML wheels.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using the local Python 3.13 installation on this machine, recreate
the environment with:

```powershell
& "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## How to Run

Option A: run the reproducible scripts:

```bash
python scripts/audit_dataset.py --raw-dir data/raw --json-out data/processed/dataset_audit.json
python scripts/run_pipeline.py --stage smoke
python scripts/run_pipeline.py --stage all --k-core 5
python scripts/create_presentation.py
python -m pytest
```

The `smoke` stage is the fast correctness check. The full `all` stage rebuilds graph features, trains the optional XGBoost/LightGBM models, regenerates figures, and can take roughly 45-60 minutes on a laptop-scale CPU runtime.

Option B: run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_network_construction.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_modeling_and_evaluation.ipynb`

Option C: run the offline Streamlit dashboard demo:

```powershell
.\.venv\Scripts\pip.exe install -r requirements-app.txt
.\.venv\Scripts\streamlit.exe run app\app.py
```

The dashboard opens at <http://localhost:8501>. It reads the exported CSV/PNG
artifacts and does not retrain models, so it is suitable for classroom
presentation. See `app/README.md` for the demo flow.

The main reusable code is in `src/`:

- `phase1.py`: loading, cleaning, filtering, splitting.
- `phase2.py`: graph construction and feature engineering.
- `phase3.py`: temporal split, baselines, models, threshold tuning, evaluation.
- `reporting_artifacts.py`: threshold, robustness, and error-analysis artifacts.
- `visualization.py`: report-ready figures.

Supporting folders:

- `reports/`: saved figures and report-facing outputs.
- `docs/final_report.md`: full paper-style report.
- `docs/final_presentation.pptx`: final slide deck.
- `models/`: optional trained-model artifacts for later inference extensions.

## Important Limitations

- `LINK_SENTIMENT` is a derived label, not a perfect ground-truth label of real-world conflict.
- A negative hyperlink is a proxy for negative inter-community interaction, not direct proof of raids or harassment.
- K-core filtering is useful for course-scale modeling, but the report should state that it restricts evaluation to a denser subgraph.
- Strict temporal evaluation is used to reduce leakage: no future label-window information is used as a model feature.

## Team Members

| No. | Full Name | Class | Student ID | GitHub | Tasks | Progress |
| :--: | :-- | :--: | :--: | :-- | :-- | :--: |
| 1 | Tran Viet Gia Huy | CS0001 | 31231027056 | [@Tommyhuy1705](https://github.com/Tommyhuy1705) | - Network Construction and Feature Engineering<br>- Build the signed directed MultiDiGraph using NetworkX<br>- Extract node, pair, community, structural balance, and text-based features<br>- Analyze network structure and community-level negative patterns<br>- Write report<br>- Slides | 100% |
| 2 | Nguyen Minh Nhut | CS0001 | 31231022656 | [@Sura3607](https://github.com/Sura3607) | - Modeling, Evaluation, and Robustness Analysis<br>- Implement anti-leakage temporal split for model evaluation<br>- Train baseline models, Logistic Regression, Random Forest, XGBoost, and LightGBM<br>- Perform ablation study, threshold tuning, robustness analysis, and error analysis<br>- Write report<br>- Slides | 100% |
| 3 | Nguyen Trong Huong | CS0001 | 31231023691 | [@trongjhuongwr](https://github.com/trongjhuongwr) | - Dashboard, Pipeline, and Final Reporting<br>- Build the Streamlit dashboard with five interactive tabs<br>- Implement Plotly charts and data access layer for dashboard visualization<br>- Maintain pipeline scripts, unit tests, and presentation generation script<br>- Write report<br>- Slides | 100% |
| 4 | To Xuan Dong | CS0001 | 31231025345 | [@xuandongg1801](https://github.com/xuandongg1801) | - Data Engineering and Exploratory Data Analysis<br>- Load, clean, and merge Reddit hyperlink datasets<br>- Perform k-core filtering and dataset audit<br>- Analyze label distribution, temporal trends, and top negative sources/targets<br>- Write report<br>- Slides | 100% |

## License

This project is for educational use in the UEH Social Media Data Analysis course.
