# MaterialMind AI — Step 3 Implementation & Audit Report

**Academic Level:** Third-Year Computer Engineering Project  
**Task Scope:** Step 3 — Material Substitution, Search Algorithms, Constraint Satisfaction & Cost Comparison  
**Audit Date:** 2026-11-19  
**Status:** **100% Implemented, Evaluated, and Tested (12/12 Tests Passing)**

---

## 1. Introduction

In Step 3, we implemented the complete core reasoning, search, and material substitution recommendation engine for **MaterialMind AI**. This system enables engineers and students to explore alternatives to a baseline steel material under user-defined mechanical property requirements, compositional limits, and budgetary constraints.

The system combines:
1. Classical Artificial Intelligence principles (**Constraint Satisfaction Problems** and **Graph Search**).
2. Supervised Machine Learning (incorporating predictions from Step 2 models when measured properties are missing).
3. Metallurgical domain rules (preventing cross-contamination of incompatible alloy families or operating temperatures).
4. Transparent cost estimation tracking with explicit provenance documentation.

---

## 2. Objective of Material Substitution

The primary goal is preliminary engineering screening: identifying candidate steel grades capable of replacing a baseline alloy (e.g. replacing high-cost or supply-constrained AISI 4140 or AISI 4340) while satisfying:
- Minimum tensile strength requirements.
- Minimum yield / proof stress thresholds.
- Minimum acceptable ductility or elongation reserves.
- Allowable composition boundaries (e.g. maximum carbon, minimum chromium).
- Cost constraints, where pricing data is available.

---

## 3. Dataset Sources and Strict Independence

The engine strictly enforces dataset separation:

1. **Steel Database (`data/processed/steel_db_features.csv` — 3,234 records):**
   - Covers 228 commercial AISI/ASTM/SAE steel grades tested at ambient conditions across varying section sizes and heat treatments.
   - Primary dataset for structural steel substitution, clustering, and room-temperature mechanical property matching.
2. **MatNavi Low-Alloy Steels (`data/processed/matnavi_features.csv` — 914 clean records):**
   - Covers 95 research alloys tested across elevated temperature conditions (27°C to 650°C).
   - Dedicated exclusively to high-temperature substitution queries where operating temperature must match within $\pm 25^\circ\text{C}$.

**Rule Enforced:** The substitution engine **never merges or mixes records across datasets**. Each substitution query is executed within the appropriate domain.

---

## 4. Material Similarity Methodology

Implemented in [`src/material_similarity.py`](file:///d:/Antigravity/Aiml_Project/src/material_similarity.py).

### 4.1 Chemical Composition Similarity ($S_{comp}$)
Measures how closely two alloys resemble each other in chemical formulation using normalized Euclidean distance across available elemental concentrations (wt%):
$$d_{comp} = \|\mathbf{x}_A - \mathbf{x}_B\|_2$$
$$S_{comp} = \frac{1}{1.0 + d_{comp}} \in (0, 1]$$

### 4.2 Mechanical Property Similarity ($S_{mech}$)
Measures the average fractional divergence across measured mechanical properties:
$$d_{mech} = \frac{1}{K} \sum_{k=1}^K \frac{|P_A^{(k)} - P_B^{(k)}|}{\max(|P_A^{(k)}|, |P_B^{(k)}|, 10^{-5})}$$
$$S_{mech} = \frac{1}{1.0 + d_{mech}} \in (0, 1]$$

### 4.3 Material Similarity Graph
- **Nodes ($V$):** Each node represents an alloy test record ($3,234$ nodes in Steel DB).
- **Edges ($E$):** An undirected edge is formed when $S_{comp} \ge 0.60$ (yielding $12,103$ bidirectional edges).
- **Edge Transition Cost:**
  $$\text{cost}(u, v) = 1.0 - S_{comp}(u, v)$$
  Lower edge cost signifies minimal chemical friction when transitioning between alloy compositions.

---

## 5. Constraint Satisfaction Problem (CSP) Formulation

Implemented in [`src/constraint_satisfaction.py`](file:///d:/Antigravity/Aiml_Project/src/constraint_satisfaction.py).

### 5.1 Hard vs. Soft Constraints
- **Hard Constraints (Mandatory Elimination):** Must be strictly satisfied; failure immediately disqualifies the candidate.
- **Soft Preferences (Ranking):** Used only after CSP filtering to rank the surviving feasible candidates.

### 5.2 Active Hard Constraints
1. **$C_1$ (Tensile Strength):** $TS \ge \text{required\_min\_ts}$.
2. **$C_2$ (Yield Strength):** $YS \ge \text{required\_min\_ys}$ (if specified).
3. **$C_3$ (Ductility):** $\text{Ductility} \ge \text{required\_min\_ductility}$ (if specified).
4. **$C_4$ (Cost Limit):** $\text{Cost} \le \text{max\_acceptable\_cost}$ (when pricing exists, or when `cost_mandatory=True`).
5. **$C_5$ (Composition Limits):** $C \le \text{max\_carbon}$, $Cr \ge \text{min\_chromium}$.
6. **$C_6$ (Operating Temperature):** $|Temp - \text{operating\_temp}| \le 25^\circ\text{C}$ (for MatNavi queries).

**Core Safety Principle:** Missing properties **never** satisfy a constraint. If a property is missing (`NaN`), the candidate is disqualified with an explicit reason logged in `rejected_records`.

---

## 6. Breadth-First Search (BFS) Implementation

Implemented in [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py).

- **Mechanism:** Explores the material similarity graph using a FIFO queue (`collections.deque`).
- **Working:** Starts at the reference alloy (Level 0), checks direct nearest neighbors (Level 1), then explores Level 2 (alloys reachable through one intermediate composition), stopping at `max_depth=2`.
- **Engineering Role:** Guarantees finding all reachable goal materials with the fewest degrees of chemical substitution hops.
- **Verification:** On a query from AISI 4140, BFS visited 10 nodes across 2 levels and identified 4 qualifying goal materials.

---

## 7. Depth-First Search (DFS) Implementation

Implemented in [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py).

- **Mechanism:** Explores deeply along an alloy branch using a LIFO stack before backtracking.
- **Working:** Follows an edge as deeply as possible up to `max_depth=3`, maintaining a `visited` set to prevent cycles.
- **Engineering Role:** Simulates deep sequential alloy modifications (e.g. low-carbon -> medium-carbon -> alloyed Cr-Mo -> Ni-Cr-Mo).
- **Verification:** Explored 9 nodes deeply and terminated without infinite loops.

---

## 8. A* Search (Heuristic-Guided Search) Implementation

Implemented in [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py).

- **State Evaluation:** $f(n) = g(n) + h(n)$
  - **$g(n)$:** Sum of edge transition costs $\sum (1.0 - S_{comp})$.
  - **$h(n)$:** Property deficit heuristic:
    $$h(n) = \max\left(0, \frac{\text{Required\_TS} - TS(n)}{\text{Required\_TS}}\right) + \max\left(0, \frac{\text{Required\_Duct} - \text{Duct}(n)}{\text{Required\_Duct}}\right)$$
    When both property thresholds are met, $h(n) = 0$.
- **Heuristic Characterization:** Documented formally as **heuristic-guided search**, providing guiding pressure toward stronger materials without overclaiming global graph optimality.
- **Verification:** Starting at AISI 4140 (Node 20), A* expanded only 2 nodes and found an optimal transition path directly to Node 25 (an oil-quenched high-strength AISI 4140 variant with $TS = 965\text{ MPa}$) with path cost $g = 0.0000$.

---

## 9. Local Search (Hill-Climbing) Implementation

Implemented in [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py).

- **Objective Function:**
  $$\text{Score}(M) = 0.40 \cdot S_{comp}(M, R) + 0.35 \cdot S_{mech}(M, R) + 0.15 \cdot \text{DuctScore}(M) - 0.10 \cdot \text{CostPenalty}(M)$$
- **Working:** Starts at an initial candidate and inspects all adjacent neighbors in the similarity graph. Moves to the neighbor with the highest score until no neighbor improves the score (local optimum).
- **Verification:** Starting at an initial candidate (Node 25, score 0.7198), Local Search climbed to Node 20 (score 0.8043) and halted at a local maximum in 1 step.

---

## 10. Candidate Ranking Methodology

Implemented in [`src/material_substitution.py`](file:///d:/Antigravity/Aiml_Project/src/material_substitution.py).

Surviving feasible candidates are ranked using a multi-criteria score:
$$\text{Suitability Score} = 0.35 \cdot S_{comp} + 0.35 \cdot S_{mech} + 0.15 \cdot \text{DuctMargin} + 0.15 \cdot \text{CostAdvantage}$$
- **$S_{comp}$ (35%):** How closely the chemical formulation matches the reference material.
- **$S_{mech}$ (35%):** Closeness of mechanical property profile.
- **$\text{DuctMargin}$ (15%):** Ductility reserve ($\min(1.0, \text{Ductility}/30\%)$).
- **$\text{CostAdvantage}$ (15%):** Relative savings compared to reference material cost.

Ranking is 100% deterministic: identical queries yield identical candidate orderings.

---

## 11. Cost Estimation Methodology

Implemented in [`src/cost_estimation.py`](file:///d:/Antigravity/Aiml_Project/src/cost_estimation.py).

- **Three-Tier Provenance System:**
  1. **User-Provided Quote:** If the user enters a specific price quote, it takes precedence and is labeled `"User-entered current market price quotation"`.
  2. **Illustrative Reference Estimate:** Looks up reference ranges in `data/raw/illustrative_supplier_reference.csv` (e.g. AISI 4140 = $0.80–$1.20/kg) and labels: `"Illustrative reference estimate — not verified transaction price (2024-01)"`.
  3. **Price Unavailable:** If no reference or user price exists, returns `"Price unavailable"`.
- **Rule Enforced:** No prices are fabricated or cross-substituted. Scraped supplier data remains logged as **0 records / Pending Permitted Source**.

---

## 12. Testing and Validation Results

Executed via [`src/test_substitution.py`](file:///d:/Antigravity/Aiml_Project/src/test_substitution.py).

| Test ID | Scenario Tested | Result | Verification Notes |
|---|---|---|---|
| **Test 1** | Standard Valid Substitution Query | **PASSED** | Returned structured recommendation card with safety disclaimer |
| **Test 2** | Several Alternatives Query | **PASSED** | Retrieved 5 candidates sorted in strict descending suitability order |
| **Test 3** | Narrow/Single Alternative Query | **PASSED** | Handled tight constraint window, returning qualified candidates |
| **Test 4** | Over-constrained Query (No Alternatives) | **PASSED** | Safely returned 0 candidates and provided exact rejection reasons |
| **Test 5** | Missing Mechanical Properties | **PASSED** | Records with `NaN` property strictly rejected by CSP |
| **Test 6** | Missing Price Information | **PASSED** | Returned `"Price unavailable"`; respected mandatory cost toggle |
| **Test 7** | Strict Cost Limit Enforcement | **PASSED** | All recommended substitutes complied with the $\le \$0.70/\text{kg}$ cap |
| **Test 8** | Strict Strength Requirement | **PASSED** | Every substitute satisfied $TS \ge 1100\text{ MPa}$ and $YS \ge 900\text{ MPa}$ |
| **Test 9** | Dataset Separation & Temperature | **PASSED** | MatNavi queries remained isolated; temperature matched within $\pm 25^\circ\text{C}$ |
| **Test 10** | Unreachable Graph Node | **PASSED** | Graph search terminated gracefully on disconnected mock components |
| **Test 11** | BFS, DFS, A*, Local Search Behavior | **PASSED** | All 4 algorithms terminated deterministically on real steel graph |
| **Test 12** | Determinism & Step 1/2 Integrity | **PASSED** | Results 100% deterministic; all Step 1 and Step 2 artifacts intact |

**Overall Test Suite Result: 12 Passed, 0 Failed (Execution time: 14.3 seconds).**

---

## 13. System Limitations

1. **Microstructure & Heat Treatment Dependence:** While the system recommends materials with matching recorded processing conditions, real fabrication outcomes depend on section size, hardenability depth ($Jominy$ curve), and heat treatment furnace control.
2. **Missing High-Cycle Fatigue & Impact Data:** Neither raw dataset contains Charpy V-notch impact toughness (Joules) or fatigue endurance limits ($S-N$ curves).
3. **Price Freshness:** Illustrative supplier reference values reflect approximate January 2024 educational data, not live spot-market quotations.

---

## 14. Conclusion & Safety Disclaimer

Step 3 successfully integrates Computer Science search algorithms, constraint satisfaction, machine learning model predictions, and metallurgical domain rules into a practical, transparent decision-support system.

> **MANDATORY SAFETY DISCLAIMER:**  
> *This material substitution recommendation is generated strictly for educational and preliminary screening decision-support. Real-world engineering substitution requires certified laboratory qualification, metallurgical microstructure verification, fatigue and fracture toughness testing, corrosion assessment, and full compliance with governing engineering standards (e.g. ASTM, ASME, ISO, SAE).*
