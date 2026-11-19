# MaterialMind AI — Final Project-Wide Debugging, Integration Repair & Runtime Validation Report

**Project Title:** MaterialMind AI: An Intelligent Machine Learning System for Mechanical Property Prediction, Cost Estimation and Material Substitution in Low-Alloy Steels  
**Academic Level:** Third-Year Computer Engineering Capstone Project  
**Date of Audit & Repair:** November 2026  
**Final Status:** **ALL DISCOVERED INTEGRATION & RUNTIME DEFECTS RESOLVED AND VERIFIED (100% OPERATIONAL)**

---

## 1. Original Error

When launching the Streamlit application via `streamlit run app.py`, the application halted immediately with the following traceback:

```text
ImportError: cannot import name 'RAW_DIR' from 'model_utils'
File "D:\Antigravity\Aiml_Project\app.py", line 27
```

---

## 2. Root Cause of Original Error

In `src/model_utils.py`, directory path constants were defined for:
- `PROJECT_ROOT` (`.../Aiml_Project`)
- `PROCESSED_DIR` (`.../data/processed`)
- `MODELS_DIR` (`.../models`)
- `ASSETS_DIR` (`.../assets`)

However, `RAW_DIR` (`.../data/raw`) was omitted from `src/model_utils.py`. In `app.py`, line 27 imported `RAW_DIR` alongside `PROCESSED_DIR`, `MODELS_DIR`, and `ASSETS_DIR`. Because `RAW_DIR` was not exported by `model_utils.py`, Python raised an `ImportError`.

---

## 3. Files Inspected

During this project-wide audit, the following files were inspected in detail:
1. `app.py`
2. `src/model_utils.py`
3. `src/cost_estimation.py`
4. `src/material_substitution.py`
5. `src/material_similarity.py`
6. `src/constraint_satisfaction.py`
7. `src/search_algorithms.py`
8. `src/regression_models.py`
9. `src/classification_models.py`
10. `src/ensemble_models.py`
11. `src/clustering_models.py`
12. `src/dimensionality_reduction.py`
13. `src/train_all.py`
14. `src/test_substitution.py`
15. `src/data_preprocessing.py`
16. `src/feature_engineering.py`
17. `src/data_analysis.py`
18. `src/web_scraping.py`
19. `data/processed/steel_db_features.csv`
20. `data/processed/matnavi_features.csv`
21. `data/processed/regression_model_comparison.csv`
22. `data/processed/classification_model_comparison.csv`
23. `data/processed/clustering_model_comparison.csv`
24. `data/raw/illustrative_supplier_reference.csv`
25. `requirements.txt`
26. `README.md`
27. `FINAL_PROJECT_REPORT.md`
28. `STEP_2_VALIDATION_REPORT.md`

---

## 4. All Errors & Inconsistencies Discovered

| # | Error / Inconsistency | Location | Root Cause | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | `ImportError: cannot import name 'RAW_DIR' from 'model_utils'` | `app.py:27` & `model_utils.py` | `RAW_DIR` was not defined in `model_utils.py` path constants. | **FIXED AND VERIFIED** |
| **2** | `AttributeError: 'CostEstimator' object has no attribute 'get_all_reference_materials'` | `app.py:1095` & `cost_estimation.py` | Method was called in Section 6 of `app.py` but was not implemented on `CostEstimator` (which stores `self.reference_df`). | **FIXED AND VERIFIED** |
| **3** | `KeyError: 'material_name'` in Cost Comparison selection | `app.py:1115` | The column in `illustrative_supplier_reference.csv` is named `'grade'`, not `'material_name'`. | **FIXED AND VERIFIED** |
| **4** | Missing `sys.path` resilience for internal imports | `src/*.py` modules | When imported outside `src/`, modules did not guarantee `src/` was in `sys.path`. | **FIXED AND VERIFIED** |
| **5** | Recommendation card property deltas key mismatch | `app.py:838` | Card dictionary keys are `delta_tensile_mpa` and `delta_yield_mpa`, while `app.py` looked for `delta_tensile_strength_mpa`, showing `"N/A"`. | **FIXED AND VERIFIED** |
| **6** | Claiming A* path as "Optimal Pathway" | `app.py:1067` | Non-linear metallurgical property deficits cannot be proven strictly admissible; should be labeled "Heuristic-Guided Pathway". | **FIXED AND VERIFIED** |
| **7** | K-Means ARI = 1.0000 mischaracterization | `app.py`, `README.md`, `FINAL_PROJECT_REPORT.md` | ARI = 1.0000 is the mathematical self-agreement benchmark ($\text{ARI}(K, K) = 1.0$). True inter-algorithm consensus is $\text{ARI}(K, Hierarchical) = 0.8622$. Comparison against historical Excel labels is $\text{ARI} = 0.0051$ (confirming orthogonal heat-treatment grouping). | **FIXED AND VERIFIED** |
| **8** | MatNavi dataset record count ambiguity | `app.py`, `README.md`, `FINAL_PROJECT_REPORT.md` | 915 raw records in dataset, 914 clean records after filtering outlier row 626 (typo $TS = 6,661\text{ MPa}$). | **FIXED AND VERIFIED** |
| **9** | Trained model count under-reporting | `app.py`, `README.md`, `FINAL_PROJECT_REPORT.md` | Previously reported 28 models. Actual count: 20 regression + 7 classification + 2 clustering + 1 PCA = **30 trained Joblib models** (+ 30 paired metadata JSONs = 60 files). | **FIXED AND VERIFIED** |

---

## 5. Files Modified & Exact Corrections Made

### 1. `src/model_utils.py`
- **Correction:** Defined `RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")` alongside `PROCESSED_DIR`, `MODELS_DIR`, and `ASSETS_DIR`.
- **Status:** **FIXED AND VERIFIED.**

### 2. `src/cost_estimation.py`
- **Correction:** Added the `get_all_reference_materials()` method:
  ```python
  def get_all_reference_materials(self):
      """Return a copy of the reference materials table."""
      return self.reference_df.copy()
  ```
- **Status:** **FIXED AND VERIFIED.**

### 3. `app.py`
- **Correction 1:** In Section 6 (Cost Comparison), updated grade selection to dynamically handle `'grade'` or `'material_name'`:
  ```python
  grade_col = "grade" if "grade" in ref_table.columns else "material_name"
  grade_options = ref_table[grade_col].tolist()
  ```
- **Correction 2:** In Section 4 (Material Substitution), corrected recommendation delta keys to match `MaterialSubstitutionEngine` output:
  ```python
  "Δ TS vs Ref": deltas.get("delta_tensile_mpa", "N/A"),
  "Δ YS vs Ref": deltas.get("delta_yield_mpa", "N/A"),
  ```
- **Correction 3:** In Section 5 (Search Algorithms), updated A* tab and display text from "Optimal Pathway Found" to **"Heuristic-Guided Pathway Found"**.
- **Correction 4:** Updated all KPI metric cards and labels to reflect verified counts: **30 trained models** and **915 MatNavi records (914 clean)**.
- **Correction 5:** Updated Section 8 Q1 viva explanation to accurately explain the K-Means self-benchmark ARI = 1.0000, inter-algorithm consensus ARI = 0.8622, and historical Excel label comparison ARI = 0.0051.
- **Status:** **FIXED AND VERIFIED.**

### 4. `src/material_substitution.py`, `src/regression_models.py`, `src/classification_models.py`, `src/clustering_models.py`, `src/ensemble_models.py`, `src/dimensionality_reduction.py`
- **Correction:** Added automatic `sys.path` insertion for `CURRENT_DIR` to guarantee safe imports regardless of whether modules are imported from the workspace root or from within `src/`.
- **Status:** **FIXED AND VERIFIED.**

### 5. `src/verify_dashboard_logic.py`
- **Correction:** Created an automated end-to-end regression testing script covering the backend execution logic of all 8 Streamlit dashboard sections.
- **Status:** **FIXED AND VERIFIED.**

### 6. `README.md` & `FINAL_PROJECT_REPORT.md`
- **Correction:** Synchronized documentation to reflect:
  - Exact model inventory: **30 trained models** (20 regression, 7 classification, 2 clustering, 1 PCA) paired with 30 JSON metadata files.
  - MatNavi record counts: **915 total records**, **914 clean training/testing records** (outlier row 626 excluded).
  - Clarified K-Means ARI = 1.0000 self-identity benchmark vs ARI = 0.8622 consensus vs historical label ARI = 0.0051.
  - A* search designated strictly as heuristic-guided search.
- **Status:** **FIXED AND VERIFIED.**

---

## 6. Verification Test Results

### 6.1 Python Compilation Check (`compileall`)
```powershell
python -m compileall -q .
```
- **Exit Code:** `0`
- **Errors / Warnings:** None.
- **Result:** **FIXED AND VERIFIED.**

### 6.2 Module Import Audit
All 18 modules in `src/` and `app.py` imported successfully with zero errors:
- `classification_models` — OK
- `clustering_models` — OK
- `constraint_satisfaction` — OK
- `cost_estimation` — OK
- `data_analysis` — OK
- `data_preprocessing` — OK
- `dimensionality_reduction` — OK
- `ensemble_models` — OK
- `feature_engineering` — OK
- `material_similarity` — OK
- `material_substitution` — OK
- `model_utils` — OK
- `regression_models` — OK
- `search_algorithms` — OK
- `test_substitution` — OK
- `train_all` — OK
- `verify_dashboard_logic` — OK
- `web_scraping` — OK
- **Result:** **FIXED AND VERIFIED.**

### 6.3 Machine Learning Model Loading & Inference Test
- All **20 regression models** across 5 mechanical targets loaded and successfully executed sample inference.
- All **7 classification models & ensembles** loaded and successfully executed sample inference.
- Both **clustering models** (K-Means & Agglomerative Hierarchical) and **PCA transformer** loaded and executed successfully.
- **Result:** **FIXED AND VERIFIED.**

### 6.4 Automated Step 3 Test Suite (`test_substitution.py`)
```powershell
python src/test_substitution.py
```
- **Execution Time:** 20.415 seconds
- **Tests Executed:** 12 unit test scenarios
- **Failures / Errors:** `0`
- **Output:** `Ran 12 tests in 20.415s — OK`
- **Result:** **FIXED AND VERIFIED.**

### 6.5 End-to-End Dashboard Logic Test (`verify_dashboard_logic.py`)
```powershell
python src/verify_dashboard_logic.py
```
- **Datasets (Sec 1 & 2):** Passed (Steel DB: 3234x31, MatNavi: 915x27).
- **Property Prediction (Sec 3):** Passed (Decision Tree Regressor predicted 640.8 MPa).
- **Material Substitution (Sec 4):** Passed (Valid query: AISI 4140 -> 3 alternatives; Over-constrained query -> 0 feasible with rejection log).
- **Search Algorithms (Sec 5):** Passed (BFS, DFS, A*, Local Search executed and terminated).
- **Cost Estimation (Sec 6):** Passed (20 reference alloys loaded).
- **Model Comparison Leaderboards (Sec 7):** Passed (20 regression, 7 classification, 2 clustering evaluations verified).
- **Result:** **FIXED AND VERIFIED.**

### 6.6 Streamlit Application Startup & Probe Test
```powershell
python -m streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1
```
- **Server Binding:** Bound to `http://127.0.0.1:8501`.
- **HTTP Probe Response:** `HTTP 200 OK`.
- **Startup Tracebacks:** `0`.
- **Result:** **FIXED AND VERIFIED.**

---

## 7. Remaining Issues

- **None.** All discovered integration errors, import issues, property delta discrepancies, and documentation inconsistencies have been identified, corrected, and verified through automated tests.

---

## 8. Exact Command to Launch the Final Application

To run the application locally:

```powershell
cd d:\Antigravity\Aiml_Project
python -m streamlit run app.py
```

The application will open automatically at:
```
http://localhost:8501
```

To run the automated test suite at any time:
```powershell
python src/test_substitution.py
```

To run the full dashboard logic test:
```powershell
python src/verify_dashboard_logic.py
```
