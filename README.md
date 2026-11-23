# MaterialMind AI

**An Intelligent Machine Learning System for Mechanical Property Prediction, Cost Estimation and Material Substitution in Low-Alloy Steels.**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![Syllabus Aligned](https://img.shields.io/badge/AIML_Syllabus-Units_2%2C3%2C4%2C5%2C6-brightgreen.svg)](#syllabus-alignment-matrix)
[![Anti-Leakage Validated](https://img.shields.io/badge/Validation-GroupShuffleSplit-success.svg)](#anti-leakage-validation)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#)

---

## 📌 Project Overview
**MaterialMind AI** is an intelligent materials informatics and decision-support machine learning system developed for a B.Tech Artificial Intelligence and Machine Learning (AIML) major project. It connects metallurgy and computer science through:

1. **Mechanical Property Prediction (Unit 3, 4, 5 Syllabus):** 8 Regression algorithms predicting Tensile Strength ($R_m$), Yield Strength / Proof Stress ($R_e$), and Ductility ($A$) from chemical composition and heat treatment conditions.
2. **Strength Tier Classification (Unit 4 & 5 Syllabus):** 8 Classifiers (5 base models + 3 ensembles) categorizing steels into Low (<600 MPa), Medium (600–950 MPa), and High ($\ge$950 MPa) strength tiers.
3. **Unsupervised Clustering & PCA (Unit 6 Syllabus):** K-Means ($k=4$), Agglomerative Hierarchical Clustering, and Principal Component Analysis (PCA, 10 components, 86.02% cumulative variance) on 23 standardized chemical elements and metallurgical indices.
4. **Material Substitution Engine (Unit 2 Syllabus):** Constraint Satisfaction Problem (CSP) hard filtering and multi-objective soft ranking recommending optimal structural steel substitutes.
5. **Graph Search Demonstrator (Unit 2 Syllabus):** Classical search algorithms (**Breadth-First Search (BFS)**, **Depth-First Search (DFS)**, **A\* Heuristic Search**, and **Hill-Climbing Local Search**) traversing a 3,234-node topological material similarity network (12,103 bidirectional edges).
6. **Cost Comparison & Sourcing:** Commercial procurement decision-support with transparent provenance tracking and web scraping compliance audits.
7. **Interactive Streamlit Web Dashboard:** Modern light-theme engineering dashboard for demonstration, testing, and viva defense.

> **⚠️ Mandatory Engineering Safety Disclaimer:**  
> MaterialMind AI is developed strictly for educational screening and preliminary decision-support modeling. It does not certify materials as safe engineering replacements for critical structural, aerospace, or pressure-vessel applications. Real-world substitution requires physical coupon qualification, metallographic examination, and compliance with ASTM, ASME, or ISO standards.

---

## 🎓 Syllabus Alignment Matrix

| Syllabus Unit | Approved Algorithm | Implementation in MaterialMind AI | Model Task | Test Performance Status |
|---|---|---|---|---|
| **Unit 2: Problem Solving & Search** | Constraint Satisfaction (CSP) | `src/constraint_satisfaction.py` (`MaterialCSPFilter`) | Physical constraint elimination | Passed (100% boundary check) |
| **Unit 2: Problem Solving & Search** | Breadth-First Search (BFS) | `src/search_algorithms.py` (`breadth_first_search`) | Level-wise similarity graph search | Passed (Exhaustive shortest hops) |
| **Unit 2: Problem Solving & Search** | Depth-First Search (DFS) | `src/search_algorithms.py` (`depth_first_search`) | Deep pathing with cycle avoidance | Passed (Deterministic backtracking) |
| **Unit 2: Problem Solving & Search** | A* Heuristic Search | `src/search_algorithms.py` (`a_star_search`) | Heuristic-guided substitution search | Passed ($f(n) = g(n) + h(n)$) |
| **Unit 2: Problem Solving & Search** | Local Search / Hill Climbing | `src/search_algorithms.py` (`local_search_optimization`) | Greedy neighborhood gradient ascent | Passed (Plateau/optima detection) |
| **Unit 3: Regression Analysis** | Simple Linear Regression | `src/regression_models.py` (`ColumnTransformer` single-feat) | Single-input property baseline | Passed (CE vs TS: $R^2 = -0.063$) |
| **Unit 3: Regression Analysis** | Multiple Linear Regression | `src/regression_models.py` (`StandardScaler` + OLS) | Continuous mechanical prediction | Passed (All 23 features: $R^2 = -0.057$) |
| **Unit 3: Regression Analysis** | Polynomial Regression (Deg 2) | `src/regression_models.py` (`PolynomialFeatures` + Ridge) | Interaction feature modeling | Passed (MatNavi TS: $R^2 = 0.7896$) |
| **Unit 3: Self Study** | Ridge Regression (L2) | `src/regression_models.py` (`Ridge` regularization) | Collinear coefficient stabilization | Passed (MatNavi Proof: $R^2 = 0.8405$) |
| **Unit 3: Self Study** | Lasso Regression (L1) | `src/regression_models.py` (`Lasso` regularization) | Sparse automatic feature selection | Passed (MatNavi TS: $R^2 = 0.7742$) |
| **Unit 3: Self Study** | Elastic Net Regression | `src/regression_models.py` (`ElasticNet` L1+L2) | Balanced penalty regression | Passed (Steel DB TS: $R^2 = 0.1467$) |
| **Unit 4: Supervised Learning** | Logistic Regression | `src/classification_models.py` (`LogisticRegression`) | Multi-class strength categorization | Passed (Test Acc: 50.49%, F1: 0.494) |
| **Unit 4: Supervised Learning** | Gaussian Naive Bayes | `src/classification_models.py` (`GaussianNB`) | Probabilistic Bayes baseline | Passed (Test Acc: 38.96%, ROC: 0.658) |
| **Unit 4: Supervised Learning** | K-Nearest Neighbors (KNN) | `src/classification_models.py` (`KNeighborsClassifier`) | Distance-weighted classification | Passed (Test Acc: 60.88%, F1: 0.620) |
| **Unit 4: Self Study** | KNN Regressor | `src/regression_models.py` (`KNeighborsRegressor`) | Instance-based local property averaging | Passed (**Best for Steel DB YS & Duct**) |
| **Unit 5: Tree & Ensemble Learning** | Decision Tree Classifier | `src/classification_models.py` (`DecisionTreeClassifier`) | Interpretable Gini partition boundaries | Passed (Test Acc: 56.66%, F1: 0.578) |
| **Unit 5: Further Reading** | Decision Tree Regressor | `src/regression_models.py` (`DecisionTreeRegressor`) | Orthogonal non-linear partitioning | Passed (**Best for Steel DB & MatNavi TS**) |
| **Unit 5: Advanced Classifiers** | Support Vector Machine (SVM) | `src/classification_models.py` (`SVC` RBF kernel) | Maximum margin classification | Passed (**Best Base Classifier: 62.18%**) |
| **Unit 5: Ensemble Learning** | Bagging Classifier | `src/ensemble_models.py` (`BaggingClassifier` DT) | Variance reduction bootstrap ensemble | Passed (**Top ROC-AUC: 0.8416**) |
| **Unit 5: Ensemble Learning** | AdaBoost Classifier | `src/ensemble_models.py` (`AdaBoostClassifier` stumps) | Sequential adaptive boosting | Passed (**Top Overall Classifier: 68.18%**) |
| **Unit 5: Ensemble Learning** | Stacking Classifier | `src/ensemble_models.py` (`StackingClassifier` meta-LR) | Heterogeneous model blending | Passed (Test Acc: 62.99%, F1: 0.632) |
| **Unit 6: Unsupervised Learning** | K-Means Clustering | `src/clustering_models.py` (`KMeans` $k=4$) | Unsupervised alloy clustering | Passed (Silhouette Score: **0.8120**) |
| **Unit 6: Unsupervised Learning** | Hierarchical Clustering | `src/clustering_models.py` (`AgglomerativeClustering`) | Ward linkage dendrogram analysis | Passed (Silhouette: **0.7857**, ARI: **0.8622**) |
| **Unit 6: Unsupervised Learning** | Principal Component Analysis | `src/dimensionality_reduction.py` (`PCA`) | Linear dimensionality reduction | Passed (10 PCs: **86.02% variance**) |

*Note: Per strict syllabus requirements, non-syllabus models (SGD Regressor and Voting Classifier) have been completely removed from source code, model directories, and evaluation tables.*

---

## 🏆 Best Performing Models by Target

| Prediction Task | Winning Model | Primary Metric | Held-Out Test Score | Why It Won (Metallurgical Explanation) |
|---|---|---|---|---|
| **Steel DB Tensile Strength ($R_m$)** | **Decision Tree Regressor** | Test $R^2$ / RMSE | **$R^2 = 0.5097$**, RMSE = 221.94 MPa | Non-linear threshold effects of carbon saturation and martensitic hardenability are captured cleanly by orthogonal tree partitioning rather than linear planes. |
| **Steel DB Yield Strength ($R_e$)** | **KNN Regressor** | Test $R^2$ / RMSE | **$R^2 = 0.3638$**, RMSE = 239.83 MPa | Yield strength strongly depends on localized compositional neighborhoods and solid-solution strengthening, which instance-based distance weighting captures best. |
| **Steel DB Ductility ($A$)** | **KNN Regressor** | Test $R^2$ / RMSE | **$R^2 = 0.2382$**, RMSE = 4.92% | Elongation is noisy across processing conditions; localized metallurgical matching beats global linear ($R^2 = 0.040$) and tree ($R^2 = 0.137$) models. |
| **MatNavi Tensile Strength ($R_m$)** | **Decision Tree Regressor** | Test $R^2$ / RMSE | **$R^2 = 0.8911$**, RMSE = 40.89 MPa | Captures the steep thermal softening drop between 350°C and 550°C with exceptional precision across elevated temperatures (27°C–650°C). |
| **MatNavi Proof Stress ($R_{p0.2}$)** | **Decision Tree Regressor** | Test $R^2$ / RMSE | **$R^2 = 0.9083$**, RMSE = 40.22 MPa | Accurately models dislocation glide kinetics and thermal yield boundaries across operating temperatures. |
| **Material Strength Classification** | **AdaBoost Classifier** | Macro F1 / Accuracy | **F1 = 0.6960**, Accuracy = **68.18%** | Sequential adaptive boosting focuses sample weights on borderline alloy heats between Low and Medium strength tiers, overcoming class overlap. |
| **Classification ROC-AUC Leader** | **Bagging Classifier** | One-vs-Rest ROC-AUC | **ROC-AUC = 0.8416**, Accuracy = **66.23%** | Bootstrap aggregation over 50 decision trees minimizes variance, yielding the highest separation across class probability thresholds. |

---

## 📂 Project Directory Structure

```
MaterialMind-AI/
│
├── app.py                     # Main Streamlit Web Application (8 Sections)
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # GitHub exclusions (preserves models and data)
│
├── data/
│   ├── raw/                   # Original raw datasets (strictly untouched)
│   │   ├── steel_database_with_labelled_clusters.xlsx  (3,234 records, 31 cols)
│   │   ├── MatNavi_mechanical_properties.csv           (915 records, 20 cols)
│   │   ├── illustrative_supplier_reference.csv         (20 standard alloy grades)
│   │   └── scraped_supplier_records.csv                (0 records — pending authorized source)
│   └── processed/             # Cleaned & feature-engineered datasets
│       ├── steel_db_cleaned.csv                        (3,234 rows)
│       ├── steel_db_features.csv                       (3,234 rows, 31 cols)
│       ├── matnavi_cleaned.csv                         (915 rows)
│       ├── matnavi_features.csv                        (915 rows, 27 cols)
│       ├── regression_model_comparison.csv             (40 evaluations)
│       ├── classification_model_comparison.csv         (8 evaluations)
│       ├── clustering_model_comparison.csv             (2 evaluations)
│       └── steel_db_pca_coordinates.csv                (3,234 rows)
│
├── models/                    # 51 Trained Joblib Models & Metadata JSONs
│   ├── regression/            # 40 saved models (8 algorithms x 5 targets)
│   ├── classification/        # 8 saved models (5 base classifiers + 3 ensembles)
│   ├── clustering/            # 2 clustering models (K-Means k=4, Hierarchical k=4)
│   └── pca/                   # 1 fitted PCA transformer
│
├── src/                       # Modular Python Source Code
│   ├── data_preprocessing.py  # Cleaning, renaming, outlier flagging
│   ├── feature_engineering.py # Leakage-safe metallurgical feature creation
│   ├── data_analysis.py       # EDA visualization generation
│   ├── web_scraping.py        # Supplier data collection & robots.txt verification
│   ├── model_utils.py         # GroupShuffleSplit, evaluation metrics, artifact I/O
│   ├── regression_models.py   # 8 syllabus regression algorithms & evaluation
│   ├── classification_models.py # 5 base classifiers & evaluation
│   ├── ensemble_models.py     # Bagging, AdaBoost, Stacking ensemble architectures
│   ├── clustering_models.py   # K-Means, Agglomerative Hierarchical Clustering
│   ├── dimensionality_reduction.py # PCA 2D projection & Scree plots
│   ├── train_all.py           # Master ML training orchestrator
│   ├── cost_estimation.py     # Material pricing lookup, user overrides, provenance
│   ├── material_similarity.py # Composition & property distance, similarity graphs
│   ├── constraint_satisfaction.py # CSP hard filtering engine & rejection logging
│   ├── search_algorithms.py   # BFS, DFS, A*, and Local Search implementations
│   ├── material_substitution.py # Integrated substitution engine & explanation generator
│   ├── test_full_suite.py     # Master verification test suite (7 comprehensive tests)
│   ├── test_substitution.py   # Automated substitution test suite (12 unit tests)
│   └── test_rejection_audit.py# CSP rejection audit log tests (5 edge cases)
│
└── assets/                    # High-resolution generated charts & plots (150 DPI)
```

---

## 🚀 How to Run the Application Locally

### 1. Prerequisites & Environment Setup
Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/agarwaldevansh/MaterialMind-AI.git
cd MaterialMind-AI

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify or Retrain Models (Optional)
All 51 pre-trained model artifacts are included in the repository. To retrain all models from scratch:
```bash
python src/train_all.py
```

### 3. Run Automated Tests
```bash
# Master comprehensive test suite
python src/test_full_suite.py

# Material substitution unit tests
python src/test_substitution.py

# CSP rejection audit log unit tests
python src/test_rejection_audit.py
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.
