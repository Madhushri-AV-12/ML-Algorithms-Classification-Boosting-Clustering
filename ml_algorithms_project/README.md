# Machine Learning Algorithms: Classification, Boosting, and Clustering

This repository runs three ML workflows on the supplied CSV datasets and saves metrics, predictions, and charts.

## Datasets and algorithms

| Dataset | Task | Algorithm(s) | Key output |
|---|---|---|---|
| `data/Social_Network_Ads.csv` | Classification | Random Forest and Gradient Boosting Classifier | Accuracy, weighted precision/recall/F1, confusion matrices, predictions |
| `data/insurance_pre.csv` | Boosting regression | Gradient Boosting Regressor | MAE, RMSE, R², actual-vs-predicted plot |
| `data/Mall_Customers.csv` | Clustering | K-Means | Customer cluster assignments, silhouette-based k selection, cluster plot |

`User ID` and `CustomerID` are not used as model features. For mall segmentation, annual income and spending score are standardized before K-Means. The number of clusters is selected by the best silhouette score across k=2..8.

## Run locally

Requires Python 3.10+ (3.11 recommended).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```

The script writes model metrics and predictions to `results/` and charts to `plots/`.

## Results files

- `results/metrics_summary.csv` — all main metrics in one table
- `results/summary.json` — structured summary
- `results/*_predictions.csv` — held-out predictions
- `results/mall_customers_cluster_assignments.csv` — cluster label for each customer
- `results/mall_k_selection_scores.csv` — silhouette and inertia for k=2..8
- `plots/` — generated charts

## Important interpretation notes

- Classification/regression metrics are calculated on a held-out 20% test split with `random_state=42`.
- Clustering is unsupervised; cluster numbers are arbitrary labels, not class labels.
- Results depend on these datasets and settings. They are educational experiments, not validated production models.
- The insurance model predicts the dataset's `charges` column; it does not provide medical or insurance advice.

## Push to GitHub

Create an empty repository on GitHub (for example, `ml-algorithms-datasets`), then run these commands in a terminal from this folder:

```bash
git init
git add .
git commit -m "Add classification boosting and clustering experiments"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ml-algorithms-datasets.git
git push -u origin main
```

Replace `YOUR_USERNAME` and the repository name with your own. GitHub authentication may ask you to sign in through a browser or use a personal access token. Do not share your token or password.

## Baseline run results

The project was run on the supplied datasets in this environment. See `results/metrics_summary.csv` for the full table.

- **Social Network Ads — Random Forest:** accuracy **90.0%**, weighted F1 **0.901**.
- **Social Network Ads — Gradient Boosting Classifier:** accuracy **87.5%**, weighted F1 **0.875**.
- **Insurance — Gradient Boosting Regressor:** MAE **2482.00**, RMSE **4345.32**, R² **0.878** (charges are in the dataset's original units).
- **Mall Customers — K-Means:** selected **5 clusters**, silhouette score **0.555** (best among k=2..8 tested).

These scores are one reproducible train/test run, not a guarantee of performance on new data.
