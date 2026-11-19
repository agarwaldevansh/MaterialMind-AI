# MaterialMind AI

**An Intelligent Machine Learning System for Mechanical Property Prediction, Cost Estimation and Material Substitution in Low-Alloy Steels.**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.67-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-F7931E.svg)](https://scikit-learn.org/)
[![Anti-Leakage Validated](https://img.shields.io/badge/Validation-GroupShuffleSplit-success.svg)](#anti-leakage-validation)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)](#)

---

## 📌 Project Overview
**MaterialMind AI** is an intelligent materials informatics and decision-support machine learning system developed as a Third-Year Computer Engineering Capstone Project. It bridges materials science and computer engineering by providing:

1. **Mechanical Property Prediction:** Multi-target regression models predicting Tensile Strength ($R_m$), Yield Strength / Proof Stress ($R_e$), and Ductility ($A$) from chemical composition and heat treatment conditions.
2. **Strength Tier Classification:** Machine learning classifiers and ensemble methods categorizing steels into Low (<600 MPa), Medium (600–950 MPa), and High ($\ge$950 MPa) strength tiers.
3. **Unsupervised Clustering & PCA:** K-Means, Agglomerative Hierarchical Clustering, and Principal Component Analysis on 23 metallurgical features.
4. **Material Substitution Engine:** Constraint Satisfaction (CSP) and multi-criteria soft ranking recommending optimal structural steel substitutes.
5. **Graph Search Demonstrator:** Classical search algorithms (**Breadth-First Search**, **Depth-First Search**, **A\* Heuristic Search**, and **Local Search Hill-Climbing**) traversing a 3,234-node material similarity network.
6. **Cost Comparison & Sourcing:** Commercial decision-support with transparent provenance tracking and web scraping compliance audits.
7. **Interactive Streamlit Web Dashboard:** A clean, modern 8-section engineering application for demonstration and viva defense.

> **⚠️ Mandatory Engineering Safety Disclaimer:**  
> MaterialMind AI is developed strictly for educational and preliminary decision-support screening. It does not certify materials as safe engineering replacements for critical structural, aerospace, or pressure-vessel applications. Real-world substitution requires physical qualification, metallographic examination, and compliance with ASTM, ASME, or ISO standards.

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
├── FINAL_PROJECT_REPORT.md    # Complete 4-Stage Project Documentation & Audit
├── STEP_1_REPORT.md           # Step 1 Data Inspection & Preprocessing Report
├── STEP_2_REPORT.md           # Step 2 Machine Learning Training & Evaluation Report
├── STEP_2_VALIDATION_REPORT.md# Step 2 Anti-Leakage & ARI Validation Report
├── STEP_3_REPORT.md           # Step 3 Material Substitution Engine Report
├── MODEL_EXPLANATION.md       # Viva Guide: Machine Learning Algorithms Explained
├── ALGORITHM_EXPLANATION.md   # Viva Guide: CSP, BFS, DFS, A*, Local Search Explained
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
│       ├── regression_model_comparison.csv             (20 evaluations)
│       ├── classification_model_comparison.csv         (7 evaluations)
│       ├── clustering_model_comparison.csv             (2 evaluations)
│       └── steel_db_pca_coordinates.csv                (3,234 rows)
│
├── models/                    # 30 Trained Joblib Models & Metadata JSONs
│   ├── regression/            # 20 saved regression models (Linear, Poly, DT, SGD)
│   ├── classification/        # 7 saved classification models (DT, KNN, SVM, NB, Ensembles)
│   ├── clustering/            # 2 clustering models (K-Means k=4, Hierarchical k=4)
│   └── pca/                   # 1 fitted PCA transformer
│
├── src/                       # Modular Python Source Code
│   ├── data_preprocessing.py  # Cleaning, renaming, outlier flagging
│   ├── feature_engineering.py # Leakage-safe metallurgical feature creation
│   ├── data_analysis.py       # EDA visualization generation
│   ├── web_scraping.py        # Supplier data collection & robots.txt verification
│   ├── model_utils.py         # GroupShuffleSplit, evaluation metrics, artifact I/O
│   ├── regression_models.py   # Linear, Polynomial, Decision Tree, SGD Regressors
│   ├── classification_models.py # Decision Tree, KNN, SVM, Gaussian NB Classifiers
│   ├── ensemble_models.py     # Soft Voting, AdaBoost, Stacking Classifiers
│   ├── clustering_models.py   # K-Means, Agglomerative Hierarchical Clustering
│   ├── dimensionality_reduction.py # PCA 2D projection & Scree plots
│   ├── train_all.py           # Master ML training orchestrator
│   ├── cost_estimation.py     # Material pricing lookup, user overrides, provenance
│   ├── material_similarity.py # Composition & property distance, similarity graphs
│   ├── constraint_satisfaction.py # CSP hard filtering engine & rejection logging
│   ├── search_algorithms.py   # BFS, DFS, A*, and Local Search implementations
│   ├── material_substitution.py # Integrated substitution engine & explanation generator
│   └── test_substitution.py   # Automated test suite (12 unit test scenarios)
│
├── notebooks/                 # Exploratory Jupyter Notebooks
└── assets/                    # High-resolution generated charts & plots (150 DPI)
```

---

## 🚀 How to Run the Application Locally

### 1. Prerequisites & Environment Setup
Clone the repository and install the verified dependencies:
```bash
# Clone the repository
git clone https://github.com/your-username/MaterialMind-AI.git
cd MaterialMind-AI

# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
Run the main application:
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🖥️ Streamlit Application Interface (Light Theme Product Edition)

The redesigned product interface features a clean, modern Light Theme design system (`#FFFFFF`, `#FFF1F0`) inspired by minimalist software products, with warm pastel accents (`#E8F6EC` soft mint, `#FFF0E2` soft peach, `#F1EBFF` soft lavender), refined coral buttons (`#FF765F`), clean borders (`#EEEEEE`), and crisp light Plotly visualizations:

1. **Overview (Landing Dashboard):**
   - Soft blush hero section (`#FFF1F0`) with headline *"Smarter Material Decisions, Powered by AI."* and badge tags.
   - Pastel analytics cards highlighting real dataset metrics: 3,234 Steel DB records (Mint), 915 MatNavi records [914 clean] (Peach), 30 verified ML models (Lavender), and 4 core computational engines (White).
   - Platform capability cards, empirical metallurgical data insights (solid-solution strengthening, thermal softening kinetics, clustering consensus), and safety notices.

2. **Material Explorer:**
   - Deep inspection across Commercial Low-Alloy Steels (3,234 records) and Elevated-Temperature Creep Steels (915 records [914 clean]).
   - Live search, multi-column sorting, descriptive statistics, Plotly light distribution histograms, property scatter plots, and Pearson correlation heatmaps.

3. **AI Predictions:**
   - Multi-target inference workspace for Tensile Strength ($R_m$), Yield/Proof Stress ($R_e$), and Ductility ($A$) from chemical composition inputs.
   - Industrial standard alloy presets (AISI 4140, 1045, 4340, 1020), operating temperature slider (27°C–650°C), calculated metallurgical indices in pastel panels, AdaBoost strength classification badge, and industrial benchmark comparison chart.

4. **Material Substitution:**
   - Decision-support workspace for selecting baseline alloys and specifying hard engineering constraints ($TS_{min}, YS_{min}, Duct_{min}, Cost_{max}, C_{max}$).
   - Multi-objective preference sliders, candidate specification cards, measured vs Step 2 DT predicted properties, property deltas ($\Delta TS, \Delta YS$), cost provenance, interactive Multi-Property Radar Chart, and CSP rejection audit log.

5. **Search Intelligence:**
   - Computational intelligence tools traversing the 3,234-node topological similarity network (12,103 substitution edges).
   - Multi-algorithm benchmark table comparing BFS, DFS, A* Heuristic-Guided Search, and Local Search (Hill-Climbing) with path sequence viewers and convergence trajectories.

6. **Cost Analysis:**
   - Material sourcing economics workbench with 20 standard alloy reference benchmarks.
   - Annual Material Substitution Savings Simulator calculating annual procurement expenditure reductions.
   - Transparent Web Scraping & Compliance Audit detailing MatWeb robots.txt Disallow and MetalsDepot WAF 403 restrictions.

7. **Model Analytics:**
   - Comprehensive model validation leaderboards: 20 regression models across 5 targets, 7 classifiers & ensembles, clustering metrics, and PCA variance.
   - Interactive Plotly light comparison bar charts and high-resolution artifact gallery (confusion matrices, actual vs predicted plots, dendrograms, Scree plots).

8. **About MaterialMind AI:**
   - Platform architecture, pipeline data flow diagram, complete technology stack, dataset specifications, AI/ML methodology, and engineering scope limitations.

---

## 🧪 Testing & Model Retraining

### Run the Material Substitution Test Suite
To verify the substitution engine and search algorithms against all 12 automated unit test scenarios:
```bash
python src/test_substitution.py
```
*Expected result: 12 tests passed, 0 failures.*

### Retrain All Machine Learning Models
To retrain all 30 models from scratch using GroupShuffleSplit:
```bash
python src/train_all.py
```
*Artifacts will be regenerated in `models/` and evaluation tables updated in `data/processed/`.*

---

## 📊 Summary of Machine Learning Performance

### 1. Regression Models (Held-out Test Split)
| Dataset | Target Property | Best Algorithm | Test MAE | Test RMSE | Test $R^2$ | Train $R^2$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Steel DB** | Tensile Strength ($R_m$, MPa) | **Decision Tree Regressor** | **158.97** | **221.94** | **0.5097** | 0.6748 |
| **Steel DB** | Yield Strength ($R_e$, MPa) | **Decision Tree Regressor** | **181.99** | **260.74** | **0.2481** | 0.6042 |
| **Steel DB** | Ductility ($A$, %) | **Decision Tree Regressor** | **4.08** | **5.23** | **0.1367** | 0.4126 |
| **MatNavi** | Tensile Strength ($R_m$, MPa) | **Decision Tree Regressor** | **30.35** | **40.89** | **0.8911** | 0.9433 |
| **MatNavi** | Proof Stress ($R_{p0.2}$, MPa) | **Decision Tree Regressor** | **27.61** | **40.22** | **0.9083** | 0.9472 |

### 2. Classification Models (Strength Tiers)
| Model Architecture | Algorithm | Test Accuracy | Macro Precision | Macro Recall | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Ensemble (Best)** | **AdaBoost Classifier** | **67.21%** | **0.7020** | **0.6952** | **0.6933** |
| Ensemble | Stacking Classifier | 62.99% | 0.6438 | 0.6528 | 0.6316 |
| Base Classifier | Support Vector Machine (SVM) | 62.18% | 0.6293 | 0.6337 | 0.6308 |
| Base Classifier | K-Nearest Neighbors (KNN) | 60.88% | 0.6157 | 0.6287 | 0.6197 |
| Ensemble | Voting Classifier (Soft) | 59.42% | 0.6174 | 0.6264 | 0.5861 |
| Base Classifier | Decision Tree Classifier | 56.66% | 0.5803 | 0.5926 | 0.5780 |
| Base Classifier | Gaussian Naive Bayes | 38.96% | 0.4612 | 0.4082 | 0.3142 |

### 3. Clustering & Dimensionality Reduction
- **K-Means ($k=4$):** Silhouette Score = **0.8120**, Inertia = 40,946.29, ARI = **1.0000** (self-identity benchmark), **0.8622** (consensus with Agglomerative Hierarchical), **0.0051** (vs historical Excel processing labels).
- **Agglomerative Hierarchical ($k=4$):** Silhouette Score = **0.7857**, Ward linkage, ARI = **0.8622** (consensus with K-Means).
- **PCA (23 Features):** Top 10 components capture **86.02%** of cumulative metallurgical variance.

---

## 🌐 Deployment to Streamlit Community Cloud

To deploy this application to **Streamlit Community Cloud**:

1. **Push Repository to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "MaterialMind AI: Complete 4-Step Release"
   git remote add origin https://github.com/<your-username>/MaterialMind-AI.git
   git push -u origin main
   ```
2. **Deploy on Streamlit:**
   - Visit [share.streamlit.io](https://share.streamlit.io).
   - Sign in with your GitHub account.
   - Click **New app**.
   - Select repository: `<your-username>/MaterialMind-AI`.
   - Set branch: `main`.
   - Set main file path: `app.py`.
   - Click **Deploy!**
3. **GitHub File Limit Verification:**  
   All files in this repository have been audited. The largest file is `data/processed/steel_db_features.csv` (~1.0 MB), and all 30 models combined total less than 5 MB. The repository is 100% compliant with GitHub's 100 MB per-file limit.

---

## 🎓 Academic Viva Defense Cheat Sheet

| Question | Short Answer |
| :--- | :--- |
| **Why GroupShuffleSplit?** | Prevents data leakage. Repeated test heats of the same alloy (e.g., AISI 4140) are isolated completely into either train or test, evaluating genuine generalization to unseen chemistries. |
| **Why did Decision Tree beat Polynomial Regression?** | Degree 2 polynomial regression on 23 features creates 276 interaction terms that overfit and extrapolate wildly ($R^2 < 0$) on unseen alloy groups. Decision trees bound predictions within realistic physical ranges. |
| **What does K-Means ARI evaluate?** | ARI = 1.0000 is the mathematical self-agreement benchmark identity. Consensus between K-Means and Agglomerative Hierarchical clustering is ARI = 0.8622 (strong structural agreement on 23 composition features). Comparison against historical Excel labels yields ARI = 0.0051 (~0), proving composition clusters are orthogonal to historical heat-treatment labels. |
| **What is the IIW Carbon Equivalent formula?** | $CE = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15$. It quantifies weldability and cold-cracking risk based on element hardenability potency relative to carbon. |
| **Why is A\* heuristic documented as heuristic-guided?** | Because non-linear metallurgical interactions mean the property shortfall heuristic is not strictly admissible in a formal mathematical proof, but serves as effective domain-guided search. |
| **Why were datasets not merged?** | Steel DB contains room-temperature commercial properties; MatNavi contains elevated-temperature creep properties (27–650°C). Merging would introduce contradictory physical states. |

---

## 📄 License & Attribution
- **Institution:** Third-Year Computer Engineering Project.
- **Datasets:** Low-Alloy Steel Database & MatNavi Creep Database (NIMS Japan).
- **License:** Educational and Academic Research Use Only.
