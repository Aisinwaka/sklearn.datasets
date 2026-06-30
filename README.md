# K-Nearest Neighbor Classification — Breast Cancer Diagnosis

## Environment Setup
```
pip install -r requirements.txt
```
or
```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Dataset
**Breast Cancer Wisconsin (Diagnostic)** — built into `sklearn.datasets.load_breast_cancer()`, so no external file is needed (chosen from the three suggested options: Iris, Wine, Breast Cancer). 569 samples, 30 numeric features computed from digitized images of fine-needle-aspirate (FNA) biopsies of breast masses, binary target: malignant (0) or benign (1). No missing values. Classes are moderately imbalanced (357 benign / 212 malignant).

## Workflow
1. **EDA**: class distribution, feature distributions by diagnosis, correlation of features with target.
2. **Preprocessing**: no missing values to handle; applied `StandardScaler` (essential for KNN since it's a distance-based algorithm — unscaled features like `mean area` would dominate distance calculations over small-scale features like `mean smoothness`).
3. **Split**: 80/20 train/test, stratified to preserve class balance.
4. **k tuning**: evaluated k = 1, 3, 5, 7, 9, 11 using 5-fold cross-validation on the training set; plotted k vs. accuracy.
5. **Final model**: trained `KNeighborsClassifier` with the optimal k, evaluated on the held-out test set.
6. **New predictions**: tested the trained model on held-out sample patients and one synthetic "average" data point.

## Results

**Cross-validation (5-fold) by k:**

| k | CV Accuracy |
|---|---|
| 1 | 0.945 |
| 3 | **0.969** |
| 5 | 0.967 |
| 7 | 0.969 |
| 9 | 0.969 |
| 11 | 0.963 |

**Optimal k = 3** (lowest k among the top-performing, lowest-variance options).

**Final test-set performance (k=3):**

| Metric | Score |
|---|---|
| Accuracy | 0.983 |
| Precision | 0.973 |
| Recall | 1.000 |
| F1 | 0.986 |
| ROC-AUC | 0.984 |

**Per-class (test set, n=114):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Malignant | 1.00 | 0.95 | 0.98 |
| Benign | 0.97 | 1.00 | 0.99 |

## Key Findings
- The model achieves very strong separation between malignant and benign tumors (ROC-AUC 0.984) once features are standardized.
- **Feature scaling was essential** — KNN is purely distance-based with no internal weighting, so without scaling, large-magnitude features dominate predictions regardless of true diagnostic relevance.
- **k=3** balances bias and variance best on this dataset: k=1 is noisier (lower, more variable CV accuracy), while accuracy plateaus from k=3 onward.
- In this medical context, **recall on the malignant class** (currently 95%) is the most important metric to monitor, since a false negative (missed cancer) is far costlier than a false positive (unnecessary follow-up). If used in a real screening-support context, the classification threshold should be tuned to push malignant recall as close to 100% as possible, even at some cost to precision.

## Files
- `knn_classification.ipynb` — full notebook, all cells executed, outputs visible.
- `requirements.txt` — dependencies.
- `README.md` — this file.

(No external dataset file is included since the Breast Cancer dataset loads directly from scikit-learn.)
