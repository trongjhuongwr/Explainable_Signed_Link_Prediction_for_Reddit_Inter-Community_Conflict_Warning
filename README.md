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

The project uses the Stanford SNAP Reddit Hyperlink Network:

- `soc-redditHyperlinks-body.tsv`
- `soc-redditHyperlinks-title.tsv`

Source: <https://snap.stanford.edu/data/soc-RedditHyperlinks.html>

The raw files should be placed in `data/raw/`. They are not committed because of size.

Expected raw schema:

- `SOURCE_SUBREDDIT`
- `TARGET_SUBREDDIT`
- `POST_ID`
- `TIMESTAMP`
- `LINK_SENTIMENT`
- `PROPERTIES`

The `PROPERTIES` column contains 86 numeric text-property features. The pipeline parses these into `text_property_00` to `text_property_85` for text-only and hybrid ablation experiments.

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

The latest verified `.venv` run is summarized in `docs/run_summary.md`. The exported metrics include 41 model/feature-set rows across Logistic Regression, Random Forest, XGBoost, LightGBM, dummy baselines, and a historical negative-ratio heuristic.

Best test result by PR-AUC:

| Feature Set | Model | Test PR-AUC | Test ROC-AUC | Test F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hybrid | Logistic Regression | 0.1840 | 0.7569 | 0.2700 | 0.2050 | 0.3954 |

## Installation

Verified local environment: the repository `.venv` currently runs with Python 3.14 and the dependencies in `requirements.txt` installed. If recreating the environment from scratch on another machine, Python 3.11 or 3.12 is still a conservative choice because ML library wheels are usually most stable there.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

Run the notebooks in order:

1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_network_construction.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_modeling_and_evaluation.ipynb`

The main reusable code is in `src/`:

- `phase1.py`: loading, cleaning, filtering, splitting.
- `phase2.py`: graph construction and feature engineering.
- `phase3.py`: temporal split, baselines, models, threshold tuning, evaluation.
- `visualization.py`: report-ready figures.

Supporting folders:

- `reports/`: saved figures and report-facing outputs.
- `models/`: optional trained-model artifacts for later inference extensions.

## Important Limitations

- `LINK_SENTIMENT` is a derived label, not a perfect ground-truth label of real-world conflict.
- A negative hyperlink is a proxy for negative inter-community interaction, not direct proof of raids or harassment.
- K-core filtering is useful for course-scale modeling, but the report should state that it restricts evaluation to a denser subgraph.
- Strict temporal evaluation is used to reduce leakage: no future label-window information is used as a model feature.

## Team Members

- Tran Viet Gia Huy - 31231027056
- Nguyen Minh Nhut - 31231022656
- Nguyen Trong Huong - 31231023691
- To Xuan Dong - 31231025345

## License

This project is for educational use in the UEH Social Media Data Analysis course.
