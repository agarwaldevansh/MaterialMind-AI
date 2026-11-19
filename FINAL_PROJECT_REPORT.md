# MaterialMind AI — Final Comprehensive Project Report

**Project Title:** MaterialMind AI: An Intelligent Machine Learning System for Mechanical Property Prediction, Cost Estimation and Material Substitution in Low-Alloy Steels  
**Project Domain:** Artificial Intelligence in Materials Science (Materials Informatics)  
**Target Academic Level:** Third-Year Computer Engineering Capstone Project  
**Date of Completion:** November 2026  
**System Status:** Completed, Validated, Tested & Deployed  

---

## 1. Executive Summary

Modern engineering design requires selecting structural materials that satisfy strict mechanical performance criteria (Tensile Strength, Yield Strength, Ductility) while minimizing procurement costs and supply-chain friction. Traditional material selection relies heavily on static handbook lookups, trial-and-error physical testing, and human intuition, which can be slow and vulnerable to single-source vendor bottlenecks.

**MaterialMind AI** addresses this challenge by providing an integrated, data-driven materials informatics platform. Developed across four rigorous development stages, the system combines:
1. **Metallurgical Preprocessing & Feature Engineering:** Safe computation of standardized metallurgical indices (IIW Carbon Equivalent, Manganese-to-Carbon ratio, total alloying content) without target leakage.
2. **Machine Learning Pipeline (30 Models):** Rigorously trained and evaluated using group-based cross-validation (`GroupShuffleSplit`) to prevent alloy identity leakage across train and test sets.
3. **Material Substitution Engine:** Integrating Constraint Satisfaction Problem (CSP) hard filtering with soft multi-criteria preference ranking to recommend viable alternatives.
4. **Classical Graph Search Demonstrator:** Executing Breadth-First Search (BFS), Depth-First Search (DFS), A\* Heuristic-Guided Search, and Local Search (Hill-Climbing) over a 3,234-node material similarity graph.
5. **Commercial Cost Sourcing & Provenance Tracking:** Providing transparent cost estimates with verified compliance audits (MatWeb `robots.txt` Disallow and MetalsDepot WAF restrictions).
6. **Interactive Streamlit Web Dashboard:** A clean, responsive 8-section engineering dashboard built for real-time inference, demonstration, and viva defense.

---

## 2. Stage-by-Stage Implementation Review

### Stage 1: Data Preprocessing, Metallurgical Feature Engineering & Ethical Sourcing
- **Datasets Analyzed:**
  - `data/raw/steel_database_with_labelled_clusters.xlsx`: 3,234 records, 31 columns representing commercial low-alloy structural steels at room temperature.
  - `data/raw/MatNavi_mechanical_properties.csv`: 915 records, 20 columns representing low-alloy creep steels tested across elevated temperatures ($27^\circ\text{C}$ to $650^\circ\text{C}$).
- **Dataset Independence Principle:** The two datasets were analyzed and modeled separately. Blindly merging room-temperature commercial steels with elevated-temperature creep steels would create severe physical contradictions (e.g., an alloy tested at $600^\circ\text{C}$ exhibiting lower yield strength due to thermal softening conflated with room-temperature properties).
- **Physical Outlier Flagging:** MatNavi row index 626 (CSV row 628, Alloy `VbF` at $300^\circ\text{C}$) reported a Tensile Strength of $6,661\text{ MPa}$ while its proof stress was $519\text{ MPa}$. This was identified as a physical typographical error for $\sim 661\text{ MPa}$. It was flagged with `is_outlier = True` and safely excluded from ML training without modifying the raw data.
- **Metallurgical Feature Engineering:** Implemented in `src/feature_engineering.py`:
  - **IIW Carbon Equivalent ($CE$):** Standardized International Institute of Welding formula:
    $$CE = C + \frac{Mn}{6} + \frac{Cr + Mo + V}{5} + \frac{Ni + Cu}{15}$$
  - **Total Alloying Percentage:** $\sum (\text{all non-Fe alloying elements in wt\%})$.
  - **Manganese-to-Carbon Ratio:** $Mn / C$, quantifying solid-solution strengthening and pearlite morphology balance.
  - **$Cr + Mo$ Total:** Indicative of high-temperature hardenability and creep resistance.
- **Web Scraping & Ethical Sourcing Audit:**
  - Automated inspection of MatWeb (`https://www.matweb.com/robots.txt`) revealed `User-agent: * Disallow: /`.
  - Programmatic probing of MetalsDepot (`https://www.metalsdepot.com`) was blocked by Cloudflare WAF (`HTTP 403 Forbidden`).
  - In adherence to engineering ethics, automated scraping was documented as **pending authorized API access**, and curated public reference price ranges (January 2024 industrial estimates) were established with transparent provenance labels.

---

### Stage 2: Machine Learning Implementation, Anti-Leakage Validation & Auditing
- **Anti-Leakage Strategy:** Evaluated using `GroupShuffleSplit` (80% train, 20% test).
  - Steel DB was grouped by `name` (unique alloy grade name).
  - MatNavi was grouped by `alloy_code`.
  - Group overlap was audited: **0 groups shared between train and test**, guaranteeing that models were evaluated strictly on their ability to generalize to completely unseen alloy chemistries.
- **Model Families Trained (30 Artifacts in `models/`, 60 Total Files with Metadata):**
  1. **Regression Models (20 Models across 5 Targets):**
     - Linear Regression (Baseline)
     - Polynomial Regression (Degree 2 with Ridge $\alpha=50.0$)
     - Decision Tree Regressor (`max_depth=10`, `min_samples_split=10`)
     - Stochastic Gradient Descent (`SGDRegressor`, `loss='squared_error'`)
  2. **Classification Models (7 Models across 3 Strength Tiers):**
     - Tiers: Low ($<600\text{ MPa}$), Medium ($600\text{--}950\text{ MPa}$), High ($\ge 950\text{ MPa}$).
     - Decision Tree, K-Nearest Neighbors ($k=5$), Support Vector Machine (RBF kernel), Gaussian Naive Bayes.
     - Ensembles: Soft Voting Classifier, AdaBoost Classifier (`n_estimators=100`), Stacking Classifier (Meta-learner: Logistic Regression).
  3. **Clustering Models (2 Models, $k=4$):**
     - K-Means ($k=4$, Silhouette = 0.8120, Inertia = 40,946.29).
     - Agglomerative Hierarchical Clustering ($k=4$, Silhouette = 0.7857, Ward linkage).
  4. **Dimensionality Reduction (1 Model):**
     - Principal Component Analysis (PCA) on 23 standardized features. Top 10 components capture 86.02% variance.
- **Validation Audit Findings:**
  - **K-Means ARI Audit:** In `data/processed/clustering_model_comparison.csv`, K-Means reports `ARI_Agreement = 1.0000`. This represents the mathematical self-agreement identity $\text{ARI}(\text{kmeans}, \text{kmeans}) = 1.0000$ serving as the baseline benchmark.
  - **Inter-Algorithm Clustering Consensus (ARI = 0.8622):** Agglomerative Hierarchical clustering and K-Means independently assigned 3,234 steel alloys into 4 clusters with **86.22% chance-corrected agreement** on 23 standardized composition features.
  - **Orthogonality to Historical Labels (ARI = 0.0051):** Comparing K-Means against the historical Excel file's pre-existing cluster labels yields an ARI of only **0.0051** ($\sim 0$), confirming that our unsupervised models clustered by chemical composition, whereas historical Excel labels were grouped primarily by processing heat treatments (annealed vs. carburized vs. Q&T).
  - **Decision Tree Superiority:** Decision Tree Regressor achieved the highest held-out test performance ($R^2 = 0.5097$ on Steel DB, $R^2 = 0.8911$ on MatNavi). Polynomial regression overfitted on unseen groups ($R^2 < 0$) due to extreme cross-product terms across 23 features, demonstrating the advantage of orthogonal axis splits for tabular materials data.
  - **AdaBoost Ensemble Superiority:** AdaBoost achieved the highest classification accuracy (**67.21%**) and Macro F1 (**0.6933**), successfully outperforming single models by focusing on hard-to-classify transition alloys.

---

### Stage 3: Material Substitution Engine, Search Algorithms & Automated Testing
- **Core Modules Developed:**
  - `src/cost_estimation.py`: Material cost lookups, user quote overrides, provenance tracking (`user_override`, `illustrative_estimate`, `unavailable`).
  - `src/material_similarity.py`: Normalized Euclidean composition distance ($S_{comp}$) and mechanical property distance ($S_{mech}$).
  - `src/constraint_satisfaction.py`: CSP hard filtering with structured rejection logging (evaluates $TS_{min}$, $YS_{min}$, $Duct_{min}$, $Cost_{max}$, $C_{max}$, and temperature tolerance).
  - `src/search_algorithms.py`: Classic AI search algorithms:
    - **Breadth-First Search (BFS):** Explores material network level-by-level, guaranteeing shortest path in unweighted hops.
    - **Depth-First Search (DFS):** Explores deepest substitution branches before backtracking.
    - **A\* Heuristic Search:** Guided by $f(n) = g(n) + h(n)$, where $g(n) = 1.0 - S_{comp}$ and $h(n)$ is the normalized property deficit to target tensile strength.
    - **Local Search (Hill-Climbing):** Iterative neighborhood optimization balancing composition, mechanical properties, ductility margin, and cost.
  - `src/material_substitution.py`: Main integration engine generating structured recommendation cards, Step 2 Decision Tree model predictions, property deltas, and natural-language engineering rationales.
- **Automated Test Suite (`src/test_substitution.py`):**
  - 12 comprehensive unit test scenarios covering:
    - Sourcing & price provenance lookup.
    - Composition similarity bounds ($[0, 1]$).
    - Mechanical property similarity bounds.
    - CSP hard constraint filtering.
    - CSP rejection reason logging.
    - Breadth-First Search goal discovery.
    - Depth-First Search exploration ordering.
    - A\* heuristic calculation and pathway convergence.
    - Local Search hill-climbing optimization.
    - Multi-criteria soft preference ranking.
    - End-to-end substitution recommendation cards.
    - Non-existent reference alloy error handling.
  - **Test Suite Results:** `Ran 12 tests in 14.26s — OK (All 12 tests passed with 0 failures).`

---

### Stage 4: Streamlit Application, Integration & Final Delivery
- **Main Entry Point:** `app.py`.
- **UI Architecture:** Modern, clean, responsive dashboard using Streamlit and Plotly.
- **Performance Optimization:**
  - Data loading cached via `@st.cache_data`.
  - Model artifacts, substitution engines, and similarity graphs cached via `@st.cache_resource`.
- **Key Application Sections:**
  1. `🏠 Home Dashboard`: Executive summary, live KPI cards, interactive architecture flow, engineering disclaimer.
  2. `📊 Dataset Explorer`: Interactive dataset preview, column filtering, summary statistics, Plotly element distributions, property correlations, and data quality disclosures.
  3. `⚙️ Property Prediction`: Real-time multi-target regression inference, standard alloy presets (AISI 4140, 1045, 4340, 1020), temperature control for MatNavi, AdaBoost strength tier classification, and benchmark comparison charts.
  4. `🔄 Material Substitution`: Interactive substitution engine with configurable hard constraints, soft preference sliders, ranked candidate cards, multi-property Radar Chart, and CSP rejection breakdown.
  5. `🔍 Search Algorithms`: Educational workbench comparing BFS, DFS, A\*, and Local Search traversal paths, visited node counts, execution times, and convergence curves.
  6. `💰 Cost Comparison`: Sourcing reference table, annual substitution cost savings simulator, and transparent web scraping compliance audit.
  7. `📈 Model Performance`: Complete model leaderboards, interactive Plotly metric comparison charts, and embedded high-resolution artifact plots.
  8. `📚 About & Viva Guide`: Academic capstone details, viva defense Q&A preparation guide, known limitations, and mandatory safety notices.
- **Local Server Verification:** Tested headless via PowerShell (`StatusCode: 200 OK`).

---

## 3. Comprehensive Performance Benchmark

### 3.1 Regression Models (Held-out Test Evaluation)
*Evaluated using GroupShuffleSplit ($Group = \text{alloy name / alloy code}$).*

| Dataset | Target Property | Algorithm | Test MAE | Test RMSE | Test $R^2$ | Train $R^2$ | Train RMSE |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Steel DB** | Tensile Strength ($R_m$, MPa) | Linear Regression | 246.66 | 325.82 | -0.0567 | 0.4072 | 241.20 |
| **Steel DB** | Tensile Strength ($R_m$, MPa) | Polynomial Reg (Deg 2) | 201.83 | 374.93 | -0.3993 | 0.5974 | 198.76 |
| **Steel DB** | Tensile Strength ($R_m$, MPa) | **Decision Tree Regressor** | **158.97** | **221.94** | **0.5097** | 0.6748 | 178.64 |
| **Steel DB** | Tensile Strength ($R_m$, MPa) | Gradient Descent (SGD) | 260.12 | 345.92 | -0.1911 | 0.3585 | 250.91 |
| **Steel DB** | Yield Strength ($R_e$, MPa) | Linear Regression | 244.22 | 308.50 | -0.0526 | 0.4045 | 241.33 |
| **Steel DB** | Yield Strength ($R_e$, MPa) | Polynomial Reg (Deg 2) | 210.20 | 343.10 | -0.3020 | 0.5721 | 204.56 |
| **Steel DB** | Yield Strength ($R_e$, MPa) | **Decision Tree Regressor** | **181.99** | **260.74** | **0.2481** | 0.6042 | 196.73 |
| **Steel DB** | Yield Strength ($R_e$, MPa) | Gradient Descent (SGD) | 254.64 | 324.98 | -0.1681 | 0.3637 | 249.47 |
| **Steel DB** | Ductility ($A$, %) | Linear Regression | 4.37 | 5.52 | 0.0397 | 0.1982 | 5.18 |
| **Steel DB** | Ductility ($A$, %) | Polynomial Reg (Deg 2) | 5.83 | 8.73 | -1.4033 | 0.3633 | 4.61 |
| **Steel DB** | Ductility ($A$, %) | **Decision Tree Regressor** | **4.08** | **5.23** | **0.1367** | 0.4126 | 4.43 |
| **Steel DB** | Ductility ($A$, %) | Gradient Descent (SGD) | 4.34 | 5.48 | 0.0552 | 0.1929 | 5.19 |
| **MatNavi** | Tensile Strength ($R_m$, MPa) | Linear Regression | 46.91 | 59.85 | 0.7666 | 0.8228 | 53.17 |
| **MatNavi** | Tensile Strength ($R_m$, MPa) | Polynomial Reg (Deg 2) | 43.83 | 56.82 | 0.7896 | 0.8790 | 43.93 |
| **MatNavi** | Tensile Strength ($R_m$, MPa) | **Decision Tree Regressor** | **30.35** | **40.89** | **0.8911** | 0.9433 | 30.07 |
| **MatNavi** | Tensile Strength ($R_m$, MPa) | Gradient Descent (SGD) | 46.87 | 59.45 | 0.7697 | 0.8200 | 53.58 |
| **MatNavi** | Proof Stress ($R_{p0.2}$, MPa) | Linear Regression | 39.12 | 54.61 | 0.8310 | 0.8587 | 49.13 |
| **MatNavi** | Proof Stress ($R_{p0.2}$, MPa) | Polynomial Reg (Deg 2) | 37.84 | 53.29 | 0.8391 | 0.9238 | 36.08 |
| **MatNavi** | Proof Stress ($R_{p0.2}$, MPa) | **Decision Tree Regressor** | **27.61** | **40.22** | **0.9083** | 0.9472 | 30.04 |
| **MatNavi** | Proof Stress ($R_{p0.2}$, MPa) | Gradient Descent (SGD) | 37.76 | 52.52 | 0.8437 | 0.8571 | 49.40 |

### 3.2 Classification & Ensemble Models (Strength Tiers)
*Evaluated on held-out test split (616 unseen alloy samples).*

| Architecture | Model Name | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Train Accuracy | Train Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Base Classifier | Decision Tree Classifier | 0.5666 | 0.5803 | 0.5926 | 0.5780 | 0.6925 | 0.7108 |
| Base Classifier | K-Nearest Neighbors (KNN) | 0.6088 | 0.6157 | 0.6287 | 0.6197 | 0.7200 | 0.7348 |
| Base Classifier | Support Vector Machine (SVM) | 0.6218 | 0.6293 | 0.6337 | 0.6308 | 0.7093 | 0.7210 |
| Base Classifier | Gaussian Naive Bayes | 0.3896 | 0.4612 | 0.4082 | 0.3142 | 0.4481 | 0.3809 |
| Ensemble | Voting Classifier (Soft) | 0.5942 | 0.6174 | 0.6264 | 0.5861 | 0.7032 | 0.7184 |
| **Ensemble (Best)** | **AdaBoost Classifier** | **0.6721** | **0.7020** | **0.6952** | **0.6933** | **0.7040** | **0.7194** |
| Ensemble | Stacking Classifier | 0.6299 | 0.6438 | 0.6528 | 0.6316 | 0.7380 | 0.7535 |

---

## 4. Academic Viva Defense Preparation Guide

### Question 1: What is the significance of K-Means ARI = 1.0000 versus Hierarchical ARI = 0.8622?
**Viva Defense Answer:**
The Adjusted Rand Index (ARI) measures chance-corrected agreement between two clusterings. In our audited evaluation:
- In `clustering_model_comparison.csv`, K-Means ARI = 1.0000 is the mathematical self-agreement benchmark identity $\text{ARI}(\text{kmeans}, \text{kmeans}) = 1.0000$.
- Agglomerative Hierarchical clustering achieved an ARI of **0.8622** against K-Means. Because hierarchical clustering uses deterministic pair-wise Ward variance minimization rather than iterative centroid relocation, an ARI of 0.8622 proves strong inter-algorithm structural consensus on 23 standardized composition features.
- When evaluated against the historical raw Excel file's pre-existing cluster labels, the ARI is only **0.0051** ($\sim 0$). This definitively demonstrates that our unsupervised ML models clustered materials purely by **chemical composition**, whereas the historical Excel labels were grouped by **processing heat treatments** (annealed vs carburized vs Q&T).

### Question 2: Why was GroupShuffleSplit essential, and what would happen if standard train_test_split were used?
**Viva Defense Answer:**
In materials databases, multiple rows frequently correspond to the same alloy chemistry tested under different heat treatments or temperatures. If a standard random `train_test_split` were used, rows for alloy "AISI 4140" would be present in both the training set and the test set. The model would memorize the chemical composition of AISI 4140 and achieve artificially high test scores (data leakage).
By grouping by `name` (Steel DB) and `alloy_code` (MatNavi):
- We guaranteed that 100% of rows for any alloy were quarantined into either the train or test partition.
- Test metrics reflect genuine generalization to new, uncatalogued steel compositions.

### Question 3: Why did Polynomial Regression perform poorly on unseen groups ($R^2 < 0$) while Decision Tree succeeded?
**Viva Defense Answer:**
Expanding 23 composition features into second-degree polynomial space produces 276 interaction terms. On unseen alloy groups that occupy slightly different composition spaces, polynomial terms extrapolate with extreme polynomial divergence, producing runaway errors ($R^2 = -0.3993$). Decision trees perform orthogonal axis-aligned splits, which inherently bound predictions within the physical ranges observed in training leaves.

### Question 4: How does the A\* Heuristic function operate, and is it admissible?
**Viva Defense Answer:**
In our material similarity graph:
- **Cost $g(n)$:** Cumulative transition friction = $\sum (1.0 - S_{comp})$.
- **Heuristic $h(n)$:** Normalized tensile strength shortfall:
  $$h(n) = \max\left(0.0, \frac{TS_{target} - TS_n}{TS_{target}}\right)$$
Because metallurgical property relationships are non-linear (e.g., adding carbon increases strength non-linearly), $h(n)$ cannot be mathematically proven to never overestimate the true transition cost under all conditions. Therefore, we formally document this algorithm as **heuristic-guided search** rather than claiming strict mathematical admissibility.

### Question 5: Why did you not scrape live supplier websites for prices?
**Viva Defense Answer:**
In compliance with engineering ethics and web scraping law:
- We programmatically inspected `https://www.matweb.com/robots.txt`, which strictly disallows automated crawlers (`Disallow: /`).
- Commercial vendor MetalsDepot blocked automated queries with an HTTP 403 WAF response.
- Attempting to bypass access controls, spoof user-agents, or break CAPTCHAs violates ethical guidelines and site terms of service. Therefore, we used curated, transparently labeled public reference estimates (January 2024 industrial estimates) with full provenance tracking.

---

## 5. System Limitations & Future Scope

### Known Engineering Limitations
1. **Microstructure Blindness:** The models predict mechanical properties from bulk chemistry and gross temper temperatures. They do not have access to microscopic phase fraction data (volume % martensite, retained austenite, bainite) or prior austenite grain size ($d^{-1/2}$ Hall-Petch relationship).
2. **Dynamic & Environmental Properties:** The datasets lack dynamic fracture toughness ($K_{1c}$), Charpy V-Notch impact transition temperatures (DBTT), and high-cycle fatigue endurance limits ($S-N$ curves).
3. **Quench & Section Sensitivity:** Steel mechanical properties depend heavily on the ruling section thickness and the quench severity (Jominy end-quench hardenability).

### Future Roadmap
1. **Computer Vision Integration:** Incorporating SEM/EBSD metallographic micrographs into a multimodal deep learning pipeline.
2. **Thermodynamic Modeling Integration:** Coupling ML predictions with CALPHAD thermodynamic equilibrium software (Thermo-Calc) to simulate phase equilibria.
3. **Supplier API Partnerships:** Integrating official commercial procurement APIs (e.g. ThomasNet, Ryerson) for real-time live quotation integration.

---

## 6. Project Delivery Checklist & Verification

- [x] **Project Foundation (Step 1):** Raw datasets preserved untouched, cleaned datasets generated, feature engineering implemented without target leakage, robots.txt verified, and EDA charts created.
- [x] **Machine Learning Pipeline (Step 2):** 30 models trained using `GroupShuffleSplit`, evaluation leaderboards exported, confusion matrices and Scree plots generated, and validation audit documented.
- [x] **Material Substitution Engine (Step 3):** CSP hard filtering, graph search (BFS, DFS, A\*, Local Search), soft ranking, and 12/12 automated unit tests passing.
- [x] **Streamlit Web Application (Step 4):** Professional, responsive 8-section application (`app.py`), verified locally via headless execution (`HTTP 200 OK`).
- [x] **Repository Cleanliness:** `.gitignore` configured to exclude caches while preserving models and data; all files under 5 MB (100% GitHub compliant).
- [x] **Documentation:** Comprehensive `README.md`, `FINAL_PROJECT_REPORT.md`, `MODEL_EXPLANATION.md`, and `ALGORITHM_EXPLANATION.md`.

---
*Report generated and validated for MaterialMind AI Capstone Submission.*
