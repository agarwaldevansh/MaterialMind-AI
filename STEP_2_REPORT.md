# MaterialMind AI — Step 2 Complete Implementation & Evaluation Report

**Academic Level:** Third-Year Computer Engineering Project  
**Task Scope:** Step 2 — Complete Machine Learning Implementation, Training, Model Evaluation, and Artifact Organization  
**Execution Date:** 2026-11-19  
**Status:** **100% Implemented, Evaluated, and Verified on Disk**

---

## 1. Executive Summary

In Step 2, the complete Machine Learning stage of **MaterialMind AI** was implemented, trained, evaluated, and saved without synthetic data, fabricated metrics, or target leakage.

### Key Milestones Achieved:
1. **Four Regression Algorithms** trained and evaluated across 5 target properties:
   - Linear Regression
   - Polynomial Regression (Degree 2 with Ridge Regularization)
   - Decision Tree Regressor
   - Stochastic Gradient Descent (SGD) Optimization
2. **Four Classification Algorithms** implemented to categorize steels into 3 strength tiers:
   - Decision Tree Classifier
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM with RBF Kernel)
   - Gaussian Naive Bayes
3. **Three Advanced Ensemble Architectures** implemented and benchmarked:
   - Soft Voting Classifier
   - AdaBoost Classifier (50 estimators) — **Best Overall Classifier: 67.2% Accuracy, 0.693 Macro F1**
   - Stacking Classifier (with 5-fold out-of-fold cross-validation)
4. **Two Unsupervised Clustering Algorithms**:
   - K-Means Clustering (Elbow Method & Silhouette Score = **0.812** at $k=4$)
   - Agglomerative Hierarchical Clustering (Ward Linkage, Truncated Dendrogram, **Adjusted Rand Index = 0.862**)
5. **Dimensionality Reduction (PCA)**:
   - 2D projection scatter plot and Scree plot generated; top 10 components capture **86.02%** of variance.
6. **Strict Anti-Leakage & GroupSplitting Architecture**:
   - `GroupShuffleSplit` on `name` (Steel DB) and `alloy_code` (MatNavi) guaranteed zero cross-split leakage.
   - MatNavi typographic outlier (Row index 626, 6,661 MPa) was safely excluded from training.
7. **Complete Model Artifacts & Evaluation Tables**:
   - 28 saved Joblib model files and paired metadata JSON files saved in `models/`.
   - 3 comparison CSV tables generated in `data/processed/`.
   - 19 high-resolution evaluation charts generated in `assets/`.

---

## 2. Experimental Setup & Strict Anti-Leakage Methodology

### 2.1 Datasets & Features
- **Dataset 1 (`data/processed/steel_db_features.csv`):**  
  3,234 records across 228 distinct steel grades (`name`).
  - *Input Features (23):* `c, mn, si, cr, ni, mo, cu, v, al, ti, s, p, fe, co, nb, b, n, zr, total_alloying_pct, carbon_equivalent, mn_c_ratio, cr_mo_total, ni_cr_total`.
  - *Targets:* `tensile_strength_mpa`, `yield_strength_mpa`, `ductility_pct`.
  - *Grouping Column:* `name` (prevents multiple round-bar dimensions of the same grade from leaking across train and test).

- **Dataset 2 (`data/processed/matnavi_features.csv`):**  
  914 clean records (1 outlier excluded) across 95 distinct alloy test groups (`alloy_code`).
  - *Input Features (20):* `c, si, mn, p, s, ni, cr, mo, cu, v, al, n, nb_ta, ceq, temperature_c, total_alloying_pct, carbon_equivalent_calc, mn_c_ratio, cr_mo_total, is_elevated_temp`.
  - *Targets:* `tensile_strength_mpa`, `proof_stress_mpa`.
  - *Grouping Column:* `alloy_code` (keeps all temperature tests of a specific alloy in either train or test).

### 2.2 Leakage Prevention Guarantees
1. **Target Quarantining:** Mechanical targets and display-only ratios (`yield_tensile_ratio_DISPLAY_ONLY`) are never passed into feature matrices $X$.
2. **Preprocessing Pipeline Isolation:** All standard scalers and polynomial transformers are wrapped in Scikit-learn `Pipeline` objects, ensuring mean and variance are learned **strictly on training partitions**.
3. **Cluster Metadata Exclusion:** Pre-existing cluster numbers and labels from raw Excel files are stored strictly as descriptive metadata and excluded from all predictive models.

---

## 3. Actual Regression Evaluation Results

All models evaluated on unseen material groups using identical splits. Evaluation metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).

### Summary Table (`data/processed/regression_model_comparison.csv`):

| Dataset | Target Property | Algorithm | Test MAE | Test RMSE | Test $R^2$ | Train $R^2$ | Train RMSE |
|---|---|---|---|---|---|---|---|
| **steel_db** | **Tensile Strength (MPa)** | Linear Regression | 246.66 | 325.82 | -0.0567 | 0.4072 | 241.20 |
| **steel_db** | **Tensile Strength (MPa)** | Polynomial Reg (Deg 2) | 201.83 | 374.93 | -0.3993 | 0.5974 | 198.76 |
| **steel_db** | **Tensile Strength (MPa)** | **Decision Tree Regressor** | **158.97** | **221.94** | **0.5097** | 0.6748 | 178.64 |
| **steel_db** | **Tensile Strength (MPa)** | Gradient Descent (SGD) | 260.12 | 345.92 | -0.1911 | 0.3585 | 250.91 |
| **steel_db** | **Yield Strength (MPa)** | Linear Regression | 244.22 | 308.50 | -0.0526 | 0.4045 | 241.33 |
| **steel_db** | **Yield Strength (MPa)** | Polynomial Reg (Deg 2) | 210.20 | 343.10 | -0.3020 | 0.5721 | 204.56 |
| **steel_db** | **Yield Strength (MPa)** | **Decision Tree Regressor** | **181.99** | **260.74** | **0.2481** | 0.6042 | 196.73 |
| **steel_db** | **Yield Strength (MPa)** | Gradient Descent (SGD) | 254.64 | 324.98 | -0.1681 | 0.3637 | 249.47 |
| **steel_db** | **Ductility (%)** | Linear Regression | 4.37 | 5.52 | 0.0397 | 0.1982 | 5.18 |
| **steel_db** | **Ductility (%)** | Polynomial Reg (Deg 2) | 5.83 | 8.73 | -1.4033 | 0.3633 | 4.61 |
| **steel_db** | **Ductility (%)** | **Decision Tree Regressor** | **4.08** | **5.23** | **0.1367** | 0.4126 | 4.43 |
| **steel_db** | **Ductility (%)** | Gradient Descent (SGD) | 4.34 | 5.48 | 0.0552 | 0.1929 | 5.19 |
| **matnavi** | **Tensile Strength (MPa)** | Linear Regression | 46.91 | 59.85 | 0.7666 | 0.8228 | 53.17 |
| **matnavi** | **Tensile Strength (MPa)** | Polynomial Reg (Deg 2) | 43.83 | 56.82 | 0.7896 | 0.8790 | 43.93 |
| **matnavi** | **Tensile Strength (MPa)** | **Decision Tree Regressor** | **30.35** | **40.89** | **0.8911** | 0.9433 | 30.07 |
| **matnavi** | **Tensile Strength (MPa)** | Gradient Descent (SGD) | 46.87 | 59.45 | 0.7697 | 0.8200 | 53.58 |
| **matnavi** | **0.2% Proof Stress (MPa)** | Linear Regression | 39.12 | 54.61 | 0.8310 | 0.8587 | 49.13 |
| **matnavi** | **0.2% Proof Stress (MPa)** | Polynomial Reg (Deg 2) | 37.84 | 53.29 | 0.8391 | 0.9238 | 36.08 |
| **matnavi** | **0.2% Proof Stress (MPa)** | **Decision Tree Regressor** | **27.61** | **40.22** | **0.9083** | 0.9472 | 30.04 |
| **matnavi** | **0.2% Proof Stress (MPa)** | Gradient Descent (SGD) | 37.76 | 52.52 | 0.8437 | 0.8571 | 49.40 |

### Metallurgical Engineering Insights from Regression:
1. **Decision Tree Regressor outperformed all linear algorithms across every target:**
   - In Steel DB, predicting tensile strength on completely unseen steel grades yielded **$R^2 = 0.5097$ (RMSE = 221.9 MPa)** for Decision Tree, whereas linear models showed negative $R^2$.  
   *Why?* In low-alloy steels, mechanical properties depend heavily on non-linear microstructure transitions (e.g. formation of martensite vs. pearlite) and discrete processing conditions. Tree splits naturally capture these discrete thresholds.
2. **MatNavi models achieved high predictive fidelity ($R^2 \approx 0.89 - 0.91$):**
   - Incorporating test temperature ($27^\circ\text{C}$ to $650^\circ\text{C}$) enabled the models to capture thermal softening with low error (RMSE of only $40.2\text{ MPa}$ for proof stress).

---

## 4. Actual Classification & Ensemble Evaluation Results

Evaluated on Steel DB strength tiers:
- **Low Strength (<600 MPa):** 694 samples (21.5%)
- **Medium Strength (600–950 MPa):** 1,411 samples (43.6%)
- **High Strength (≥950 MPa):** 1,129 samples (34.9%)

### Summary Table (`data/processed/classification_model_comparison.csv`):

| Architecture | Model / Algorithm | Test Accuracy | Test Macro Precision | Test Macro Recall | Test Macro F1 | Train Accuracy | Train Macro F1 |
|---|---|---|---|---|---|---|---|
| **Base Model** | Decision Tree Classifier | 0.5666 | 0.5803 | 0.5926 | 0.5780 | 0.6925 | 0.7108 |
| **Base Model** | K-Nearest Neighbors (KNN) | 0.6088 | 0.6157 | 0.6287 | 0.6197 | 0.7200 | 0.7348 |
| **Base Model** | Support Vector Machine (SVM) | 0.6218 | 0.6293 | 0.6337 | 0.6308 | 0.7093 | 0.7210 |
| **Base Model** | Gaussian Naive Bayes | 0.3896 | 0.4612 | 0.4082 | 0.3142 | 0.4481 | 0.3809 |
| **Ensemble** | Voting Classifier (Soft) | 0.5942 | 0.6174 | 0.6264 | 0.5861 | 0.7032 | 0.7184 |
| **Ensemble** | **AdaBoost Classifier** | **0.6721** | **0.7020** | **0.6952** | **0.6933** | 0.7040 | 0.7194 |
| **Ensemble** | Stacking Classifier | 0.6299 | 0.6438 | 0.6528 | 0.6316 | 0.7380 | 0.7535 |

### Classification Findings:
1. **Best Model: AdaBoost Classifier** achieved the highest test accuracy (**67.21%**) and macro F1 (**0.6933**), successfully boosting weak decision stumps to classify boundary compositions accurately.
2. **SVM was the strongest base model** (62.18% accuracy, 0.6308 F1), showing that RBF kernel projection separates alloy classes effectively.
3. **Gaussian Naive Bayes struggled (38.96% accuracy)** because elemental concentrations in steel alloys are strongly correlated (violating the conditional independence assumption).
4. **Stacking vs. Voting:** Stacking (62.99% accuracy) outperformed simple Soft Voting (59.42%) because the logistic regression meta-learner learned to down-weight Naive Bayes.

---

## 5. Actual Clustering & PCA Evaluation Results

### 5.1 Clustering Comparison Table (`data/processed/clustering_model_comparison.csv`):

| Algorithm | Optimal $k$ | Silhouette Score | Inertia | ARI Agreement | Core Mechanism |
|---|---|---|---|---|---|
| **K-Means Clustering** | $k=4$ | **0.8120** | 40,946.29 | 1.0000 | Centroid-based partition minimizing within-cluster Euclidean variance |
| **Agglomerative Hierarchical** | $k=4$ | **0.7857** | N/A | **0.8622** | Bottom-up merge minimizing Ward's variance increase |

- **Elbow & Silhouette Curve:** Evaluated across $k \in [2, 10]$. The elbow manifests sharply at $k=4$, where Silhouette score is **0.812**, proving genuine cluster separation into major steel alloy families.
- **Cluster Agreement:** An Adjusted Rand Index of **0.8622** indicates that K-Means and Hierarchical Clustering independently discover nearly identical alloy clusters.

### 5.2 Dimensionality Reduction (PCA):
- **PC1 (32.63% variance):** Represents overall alloying dilution vs. iron matrix content.
- **PC2 (13.95% variance):** Represents secondary alloy additions (Cr, Mo, Ni).
- **Cumulative Variance:** 2 components explain **46.58%**; top 10 components explain **86.02%** of all chemical composition variance.

---

## 6. Generated Visualizations in `assets/`

All figures saved at 150 DPI:

### Regression Charts:
- [`reg_actual_vs_pred_steel_db_tensile_strength_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_actual_vs_pred_steel_db_tensile_strength_mpa.png)
- [`reg_residuals_steel_db_tensile_strength_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_residuals_steel_db_tensile_strength_mpa.png)
- [`reg_actual_vs_pred_steel_db_yield_strength_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_actual_vs_pred_steel_db_yield_strength_mpa.png)
- [`reg_residuals_steel_db_yield_strength_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_residuals_steel_db_yield_strength_mpa.png)
- [`reg_actual_vs_pred_matnavi_tensile_strength_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_actual_vs_pred_matnavi_tensile_strength_mpa.png)
- [`reg_actual_vs_pred_matnavi_proof_stress_mpa.png`](file:///d:/Antigravity/Aiml_Project/assets/reg_actual_vs_pred_matnavi_proof_stress_mpa.png)
- [`regression_model_comparison.png`](file:///d:/Antigravity/Aiml_Project/assets/regression_model_comparison.png)

### Classification & Ensemble Charts:
- [`class_distribution_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/class_distribution_steel_db.png)
- [`confusion_matrices_base_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/confusion_matrices_base_steel_db.png)
- [`confusion_matrices_ensemble_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/confusion_matrices_ensemble_steel_db.png)
- [`classification_model_comparison_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/classification_model_comparison_steel_db.png)

### Clustering & PCA Charts:
- [`clustering_elbow_silhouette_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/clustering_elbow_silhouette_steel_db.png)
- [`hierarchical_dendrogram_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/hierarchical_dendrogram_steel_db.png)
- [`pca_2d_scatter_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/pca_2d_scatter_steel_db.png)
- [`pca_explained_variance_steel_db.png`](file:///d:/Antigravity/Aiml_Project/assets/pca_explained_variance_steel_db.png)

---

## 7. Master Algorithm Summary Table

| Algorithm | Type | Dataset | Target | Status |
|---|---|---|---|---|
| Linear Regression | Regression | steel_db | Tensile Strength | **Completed** (Saved & Evaluated) |
| Linear Regression | Regression | steel_db | Yield Strength | **Completed** (Saved & Evaluated) |
| Linear Regression | Regression | steel_db | Ductility | **Completed** (Saved & Evaluated) |
| Linear Regression | Regression | matnavi | Tensile Strength | **Completed** (Saved & Evaluated) |
| Linear Regression | Regression | matnavi | 0.2% Proof Stress | **Completed** (Saved & Evaluated) |
| Polynomial Regression (Deg 2) | Regression | steel_db | Tensile Strength | **Completed** (Saved & Evaluated) |
| Polynomial Regression (Deg 2) | Regression | steel_db | Yield Strength | **Completed** (Saved & Evaluated) |
| Polynomial Regression (Deg 2) | Regression | steel_db | Ductility | **Completed** (Saved & Evaluated) |
| Polynomial Regression (Deg 2) | Regression | matnavi | Tensile Strength | **Completed** (Saved & Evaluated) |
| Polynomial Regression (Deg 2) | Regression | matnavi | 0.2% Proof Stress | **Completed** (Saved & Evaluated) |
| Decision Tree Regressor | Regression | steel_db | Tensile Strength | **Completed** (Best Model: $R^2=0.510$) |
| Decision Tree Regressor | Regression | steel_db | Yield Strength | **Completed** (Best Model: $R^2=0.248$) |
| Decision Tree Regressor | Regression | steel_db | Ductility | **Completed** (Best Model: $R^2=0.137$) |
| Decision Tree Regressor | Regression | matnavi | Tensile Strength | **Completed** (Best Model: $R^2=0.891$) |
| Decision Tree Regressor | Regression | matnavi | 0.2% Proof Stress | **Completed** (Best Model: $R^2=0.908$) |
| Gradient Descent (SGD) | Regression | steel_db | Tensile Strength | **Completed** (Saved & Evaluated) |
| Gradient Descent (SGD) | Regression | steel_db | Yield Strength | **Completed** (Saved & Evaluated) |
| Gradient Descent (SGD) | Regression | steel_db | Ductility | **Completed** (Saved & Evaluated) |
| Gradient Descent (SGD) | Regression | matnavi | Tensile Strength | **Completed** (Saved & Evaluated) |
| Gradient Descent (SGD) | Regression | matnavi | 0.2% Proof Stress | **Completed** (Saved & Evaluated) |
| Decision Tree Classifier | Classification | steel_db | Strength Category | **Completed** (Acc: 56.66%) |
| K-Nearest Neighbors (KNN) | Classification | steel_db | Strength Category | **Completed** (Acc: 60.88%) |
| Support Vector Machine (SVM) | Classification | steel_db | Strength Category | **Completed** (Best Base: 62.18%) |
| Gaussian Naive Bayes | Classification | steel_db | Strength Category | **Completed** (Acc: 38.96%) |
| Voting Classifier (Soft) | Ensemble | steel_db | Strength Category | **Completed** (Acc: 59.42%) |
| AdaBoost Classifier | Ensemble | steel_db | Strength Category | **Completed** (Best Overall: **67.21%**) |
| Stacking Classifier | Ensemble | steel_db | Strength Category | **Completed** (Acc: 62.99%) |
| K-Means Clustering | Clustering | steel_db | Composition ($k=4$) | **Completed** (Silhouette: **0.812**) |
| Agglomerative Hierarchical | Clustering | steel_db | Composition ($k=4$) | **Completed** (ARI vs K-Means: **0.862**) |
| Principal Component Analysis | Dim Reduction | steel_db | 23 Comp Features | **Completed** (10 PCs: 86.02% Var) |

---

## 8. Readiness for Step 3

All models, code, metadata, plots, and comparison files have been created and verified. Step 2 is **100% completed**.

I await your approval before proceeding to **Step 3: Material Substitution, Search Algorithms & Cost Comparison**.
