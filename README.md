# Dry Bean Variety Classification — ML Assignment 2

M.Tech (AIML) · Work Integrated Learning Programmes Division · BITS Pilani

**Live Streamlit app:** _<add your app URL here after deploying>_
**GitHub repository:** https://github.com/praveenmulesoft/dry-bean-classification-streamlit

---

## a. Problem statement

Dry beans are graded and priced by variety, but sorting them by hand is slow and
inconsistent between graders. The task here is to decide which of **seven bean
varieties** a single grain belongs to, using only **sixteen geometric
measurements** extracted from a photograph of that grain — area, perimeter, axis
lengths, eccentricity, roundness, compactness, and four derived shape factors.

This is a **multi-class classification** problem (7 classes). No image pixels are
involved: the computer-vision step has already reduced each grain to a row of
numbers, and the job is purely to learn the mapping from those numbers to a
variety label. Five classifiers are trained on the same split and compared on six
metrics to see which family of model best separates varieties whose shapes
overlap considerably.

## b. Dataset description

**Source:** [UCI Machine Learning Repository — Dry Bean Dataset (ID 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
Koklu, M. and Ozkan, I.A. (2020), *Multiclass Classification of Dry Beans Using
Computer Vision and Machine Learning Techniques*, Computers and Electronics in
Agriculture, 174, 105507.

| Property | Value |
|---|---|
| Instances | **13,611** (assignment minimum: 500 ✅) |
| Features | **16**, all numeric (assignment minimum: 12 ✅) |
| Target | `Class` — 7 bean varieties |
| Missing values | None |
| Task type | Multi-class classification |

Images of 13,611 grains from seven varieties were captured with a high-resolution
camera; each grain was segmented and reduced to 12 dimensional measurements plus
4 shape factors.

**Features.** `Area`, `Perimeter`, `MajorAxisLength`, `MinorAxisLength`,
`AspectRation`, `Eccentricity`, `ConvexArea`, `EquivDiameter`, `Extent`,
`Solidity`, `roundness`, `Compactness`, `ShapeFactor1`, `ShapeFactor2`,
`ShapeFactor3`, `ShapeFactor4`.

**Class distribution** — moderately imbalanced, which is why every metric below
is macro-averaged rather than weighted:

| Variety | Instances | Share |
|---|---|---|
| DERMASON | 3,546 | 26.1% |
| SIRA | 2,636 | 19.4% |
| SEKER | 2,027 | 14.9% |
| HOROZ | 1,928 | 14.2% |
| CALI | 1,630 | 12.0% |
| BARBUNYA | 1,322 | 9.7% |
| BOMBAY | 522 | 3.8% |

**Preprocessing.** Labels are integer-encoded with `LabelEncoder`. The data is
split **80/20, stratified**, at `random_state=42` → 10,888 train / 2,723 test
rows. Logistic Regression and kNN are wrapped in a `Pipeline` with
`StandardScaler`, since both are sensitive to feature scale (`Area` spans tens of
thousands while `Solidity` sits near 1.0). The tree-based models and GaussianNB
are trained on raw values, where scaling changes nothing. Scaling lives *inside*
the pipeline so the exact same transform is reapplied at inference — no
train/serve skew.

## c. GitHub repository link

https://github.com/praveenmulesoft/dry-bean-classification-streamlit

Repository contents:

```
dry-bean-classification-streamlit/
├── app.py                  # Streamlit application
├── train_models.py         # trains + evaluates + persists all five models
├── requirements.txt
├── README.md
├── test_data.csv           # held-out 20% split (2,723 rows) for the app
├── data/
│   └── dry_bean_full.csv   # full dataset (13,611 rows)
└── model/                  # persisted artefacts
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── random_forest.joblib
    ├── label_encoder.joblib
    └── metrics_summary.csv / .json
```

## d. Models used — comparison of evaluation metrics

All five models are trained on the **same** stratified 80/20 split
(`random_state=42`) and scored on the identical 2,723-row held-out test set.
Because the problem has seven imbalanced classes, Precision, Recall and F1 are
**macro-averaged** (each variety weighted equally, so the small BOMBAY class
matters as much as the large DERMASON one) and AUC is computed **one-vs-rest**
over predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | **0.9214** | **0.9948** | **0.9354** | **0.9321** | **0.9335** | **0.9050** |
| Decision Tree | 0.9115 | 0.9705 | 0.9258 | 0.9249 | 0.9252 | 0.8930 |
| kNN | 0.9185 | 0.9879 | 0.9343 | 0.9293 | 0.9315 | 0.9014 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Random Forest (Ensemble) | 0.9174 | 0.9942 | 0.9318 | 0.9273 | 0.9294 | 0.9000 |

Hyperparameters: Logistic Regression `max_iter=2000`; Decision Tree
`max_depth=12, min_samples_leaf=5`; kNN `n_neighbors=11, weights='distance'`;
GaussianNB defaults; Random Forest `n_estimators=120, max_depth=12,
min_samples_leaf=8`.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | The strongest model overall, and a genuinely surprising result — a linear decision boundary beats two non-linear competitors here. It tops every single metric: accuracy (0.9214), AUC (0.9948), macro-F1 (0.9335) and MCC (0.9050). The explanation is that the 16 engineered shape factors have already done the hard non-linear work; once the varieties are described in terms of compactness and roundness they become close to linearly separable, so extra model flexibility buys nothing. Its near-perfect 0.9948 AUC shows the predicted probabilities are well calibrated, not just the hard labels. Requires scaling to converge. |
| **Decision Tree** | The weakest of the four strong models (F1 0.9252, MCC 0.8930). A single tree carves the feature space with axis-parallel cuts, which is a poor fit for varieties separated by smooth, diagonal boundaries in correlated features — it needs a long staircase of splits to approximate what logistic regression draws with one line. The gap between its AUC (0.9705) and the others is the tell: a depth-12 tree yields coarse, piecewise-constant probability estimates, so it ranks candidates far less finely even when its hard predictions are respectable. Its redeeming quality is interpretability — the fitted rules can be read directly. |
| **kNN** | Very competitive (F1 0.9315, MCC 0.9014), essentially tied with Random Forest and just behind Logistic Regression. Beans of one variety genuinely do cluster together in the scaled 16-D space, which is exactly the assumption kNN makes, and `weights='distance'` lets nearer neighbours dominate the vote. The costs are practical rather than statistical: the model stores all 10,888 training rows (a 1.4 MB artefact, by far the largest per unit of accuracy), and every prediction is a fresh distance computation, so inference is the slowest of the five. It also collapses without scaling, since `Area` would otherwise drown out every other feature. |
| **Naive Bayes** | A clear outlier and the only genuinely weak model — accuracy 0.7639 and MCC 0.7154, roughly 16 F1 points behind the field. The cause is its defining assumption: GaussianNB treats every feature as conditionally independent given the class, but these features are severely redundant by construction. `Area`, `ConvexArea`, `EquivDiameter` and `Perimeter` all measure size, and `Compactness`, `roundness` and `ShapeFactor3` are algebraically related. Counting that evidence repeatedly makes the model overconfident and skews the boundary. Tellingly its AUC stays high at 0.9672 — the *ranking* of classes is broadly right, but the miscalibrated probabilities push the arg-max to the wrong variety often enough to wreck accuracy. |
| **Random Forest (Ensemble)** | Nearly matches the winner (F1 0.9294, AUC 0.9942, MCC 0.9000) and dramatically repairs the single Decision Tree it is built from — +0.4 F1 points and a jump from 0.9705 to 0.9942 AUC, since averaging 120 trees turns coarse piecewise-constant estimates into smooth ones. That is bagging doing exactly what it promises: variance reduction at no cost in bias. It is the most robust choice here, needing no feature scaling and few tuning decisions, but on this dataset the ensemble's flexibility is simply not required, so it cannot overtake the linear model. Depth and leaf size were capped to keep the artefact under 6 MB, costing about 0.004 macro-F1. |
| **Overall Winner for your dataset?** | **Logistic Regression.** It leads on all six metrics simultaneously — accuracy, AUC, precision, recall, F1 and MCC — while being the smallest model in the repository (3 KB versus 5.6 MB for the forest and 1.4 MB for kNN) and the fastest at inference. The broader lesson is that model complexity should be earned, not assumed: because the 16 shape descriptors already encode the non-linear geometry, the simplest classifier generalises best. Random Forest is the sensible runner-up and would be the safer pick if the feature set were ever replaced with raw, unengineered measurements. |

Interpreting the metrics together: **MCC** is the most trustworthy single number
on this imbalanced 7-class problem, because it accounts for all cells of the
confusion matrix and cannot be inflated by favouring the large DERMASON class —
its ordering agrees with macro-F1 throughout. The **AUC vs accuracy divergence
for Naive Bayes** (0.9672 vs 0.7639) is the most instructive contrast in the
table: a model can rank classes well and still choose badly once forced to commit
to a single label.

---

## Reproducing these results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train_models.py     # writes model/ artefacts + test_data.csv
streamlit run app.py       # opens the app at http://localhost:8501
```

`train_models.py` is fully seeded (`random_state=42` for the split and every
model), so the metric table above reproduces exactly.

## The Streamlit application

Deployed on Streamlit Community Cloud. The app **loads pre-trained models** from
`model/` and never trains at request time, which keeps it inside the free tier's
memory and startup limits.

Features:

1. **Dataset upload (CSV)** — sidebar uploader; `test_data.csv` in this repo is
   the intended input and reproduces the README table exactly.
2. **Model selection dropdown** — switch between all five trained models, plus a
   *Compare all five* toggle that scores every model on the upload at once and
   shades the best value in each column.
3. **Display of evaluation metrics** — all six metrics (Accuracy, AUC, Precision,
   Recall, F1, MCC) shown for the selected model.
4. **Confusion matrix and classification report** — a 7×7 annotated heatmap
   alongside the full per-class precision/recall/F1 breakdown.

It also previews the upload with a class-distribution chart, and offers
row-level predictions with a correct/incorrect flag as a downloadable CSV.

**Robustness.** The app validates that the uploaded CSV has a `Class` column and
that its labels are all recognised, failing with a clear message otherwise. If an
upload contains only a subset of the seven varieties, one-vs-rest AUC is
mathematically undefined, so it is reported as `n/a` with an explanation instead
of crashing.
