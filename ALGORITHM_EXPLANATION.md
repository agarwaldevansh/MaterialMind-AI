# MaterialMind AI — Search & Optimization Algorithm Explanations

**Academic Level:** Third-Year Computer Engineering / Artificial Intelligence  
**Target Audience:** Students, Viva Examiners, and Engineering Evaluators  
**Scope:** Conceptual, Mathematical, and Operational Breakdown of Search Algorithms & Constraint Satisfaction Implemented in Step 3.

---

## Table of Contents
1. [Constraint Satisfaction Problem (CSP)](#1-constraint-satisfaction-problem-csp)
2. [Breadth-First Search (BFS)](#2-breadth-first-search-bfs)
3. [Depth-First Search (DFS)](#3-depth-first-search-dfs)
4. [A* Search (Heuristic-Guided Search)](#4-a-search-heuristic-guided-search)
5. [Local Search (Hill-Climbing Optimization)](#5-local-search-hill-climbing-optimization)
6. [Multi-Criteria Ranking Methodology](#6-multi-criteria-ranking-methodology)

---

# 1. Constraint Satisfaction Problem (CSP)

### 1.1 Purpose
Eliminates all candidate steel materials that violate hard engineering, environmental, or budgetary constraints before soft-preference ranking or path exploration occurs.

### 1.2 Working Principle
A classical Constraint Satisfaction Problem is formally defined by three components:
$$\text{CSP} = \langle \mathcal{X}, \mathcal{D}, \mathcal{C} \rangle$$
1. **Variables ($\mathcal{X}$):** A candidate material $M$ and its attributes:
   $$\{TS(M), YS(M), \text{Duct}(M), \text{Cost}(M), C(M), Cr(M), Temp(M)\}$$
2. **Domain ($\mathcal{D}$):** All discrete material rows present in the active dataset ($3,234$ records in Steel DB; $914$ records in MatNavi).
3. **Constraints ($\mathcal{C}$):** Hard boolean predicates that must all evaluate to `True`:
   - $C_{TS}: TS(M) \ge \text{required\_min\_tensile}$
   - $C_{YS}: YS(M) \ge \text{required\_min\_yield}$ (when specified)
   - $C_{\text{Duct}}: \text{Duct}(M) \ge \text{required\_min\_ductility}$ (when specified)
   - $C_{\text{Cost}}: \text{Cost}(M) \le \text{max\_acceptable\_cost}$ (when specified and mandatory)
   - $C_{\text{Comp}}: C(M) \le \text{max\_carbon} \land Cr(M) \ge \text{min\_chromium}$
   - $C_{\text{Temp}}: |Temp(M) - \text{operating\_temp}| \le 25^\circ\text{C}$ (for MatNavi high-temperature records)

**Core Rule on Missing Data:** A missing required property **never** satisfies a constraint. If an alloy record has `NaN` for a mandatory required property, it is rejected with an explicit note.

### 1.3 Step-by-Step Example
- **User Requirement:** Substitute for a shaft material requiring $TS \ge 900\text{ MPa}$ and $\text{Ductility} \ge 15\%$.
- Candidate 1: AISI 1020 ($TS = 420\text{ MPa}, \text{Duct} = 25\%$) -> **Fails $C_{TS}$** ($420 < 900$). **Rejected.**
- Candidate 2: AISI 1095 ($TS = 965\text{ MPa}, \text{Duct} = 9\%$) -> **Fails $C_{\text{Duct}}$** ($9 < 15$). **Rejected.**
- Candidate 3: AISI 4140 Q&T ($TS = 965\text{ MPa}, \text{Duct} = 19\%$) -> **Satisfies all constraints.** **Admitted to Feasible Set.**

### 1.4 Complexity & Project Role
- **Time Complexity:** $O(N \cdot |\mathcal{C}|)$ where $N$ is the number of records and $|\mathcal{C}|$ is the number of active constraints. Highly efficient linear filtering.
- **Space Complexity:** $O(N)$ to store the filtered subset and rejection logs.
- **Implementation:** [`src/constraint_satisfaction.py`](file:///d:/Antigravity/Aiml_Project/src/constraint_satisfaction.py)

---

# 2. Breadth-First Search (BFS)

### 2.1 Purpose
Explores the material similarity graph level-by-level to identify all reachable substitute alloys within $d$ degrees of chemical/metallurgical separation from the reference material.

### 2.2 Working Principle
1. **Nodes:** Each node represents a distinct alloy instance in the dataset.
2. **Edges:** Undirected edges connect two materials whose composition similarity score $S_{comp} \ge \text{threshold}$ (default 0.60).
3. **Queue Mechanism:** Uses a FIFO (First-In, First-Out) queue (`collections.deque`).
4. **Exploration Order:**
   - Level 0: Reference material.
   - Level 1: Immediate nearest-neighbor alloys (direct metallurgical cousins).
   - Level 2: Alloys connected via one intermediate alloy bridge.
   - Level $d$: Halts at `max_depth` (default 2 or 3).
5. **Goal Testing:** Every dequeued node is tested against the goal condition (satisfying required strength and ductility).

```
Level 0:           [AISI 4140 (Ref)]
                         /     \
Level 1:      [AISI 4130]      [AISI 4150]
                 /     \            |
Level 2:  [AISI 4340]  [AISI 8620] [AISI 5160]
```

### 2.3 Step-by-Step Example
- Starting at AISI 4140 ($TS = 814\text{ MPa}$). Target: $TS \ge 900\text{ MPa}$.
- Enqueue AISI 4140 (depth 0).
- Pop AISI 4140; inspect neighbors: AISI 4130, AISI 4150, AISI 8640.
- Check Level 1:
  - AISI 4130 ($TS = 670\text{ MPa}$) -> Not goal, enqueue (depth 1).
  - AISI 4150 ($TS = 972\text{ MPa}$) -> **Goal Qualified!**
  - AISI 8640 ($TS = 945\text{ MPa}$) -> **Goal Qualified!**
- Terminates at max depth, reporting reachable alternatives grouped by depth.

### 2.4 Complexity & Project Role
- **Time Complexity:** $O(V + E)$ where $V \le 3,234$ nodes and $E \le 12,103$ edges.
- **Space Complexity:** $O(V)$ for the visited set and FIFO queue.
- **Implementation:** [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py)

---

# 3. Depth-First Search (DFS)

### 3.1 Purpose
Explores substitution branches deeply into specialized alloy variants before backtracking. Useful for tracing incremental alloy evolution (e.g. tracing low carbon -> medium carbon -> alloyed Cr-Mo -> Ni-Cr-Mo).

### 3.2 Working Principle
1. **Stack Mechanism:** Uses a LIFO (Last-In, First-Out) stack or recursive calls.
2. **Exploration Order:** Follows an edge as deeply as possible until a leaf node or `max_depth` limit is reached, then backtracks to explore alternate branches.
3. **Visited Guard:** Tracks visited node IDs to prevent cycles in the undirected material graph.

### 3.3 Contrast: BFS vs. DFS
| Feature | Breadth-First Search (BFS) | Depth-First Search (DFS) |
|---|---|---|
| **Data Structure** | FIFO Queue (`deque`) | LIFO Stack (`list.pop()`) |
| **Exploration Strategy** | Shallow, concentric level-by-level | Deep, branch-first dive |
| **Path Length Guarantee** | Finds the shortest path in edge hops | Does not guarantee shortest path |
| **Memory Footprint** | Proportional to branch width ($O(b^d)$) | Proportional to search depth ($O(b \cdot d)$) |
| **Use in MaterialMind AI** | Finds closest direct alloy analogues | Traces incremental chemical modification paths |

### 3.4 Complexity & Project Role
- **Time Complexity:** $O(V + E)$.
- **Space Complexity:** $O(d)$ where $d$ is maximum tree depth.
- **Implementation:** [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py)

---

# 4. A* Search (Heuristic-Guided Search)

### 4.1 Purpose
Navigates the material similarity graph to find a path from the reference alloy to an acceptable substitute that minimizes cumulative substitution friction while being steered by property requirements.

### 4.2 Mathematical Formulation
A* evaluates each node $n$ using the cost function:
$$f(n) = g(n) + h(n)$$
Where:
1. **Path Cost $g(n)$:** The accumulated transition cost from the start node:
   $$g(n) = \sum_{(u, v) \in \text{path}} \text{edge\_cost}(u, v) = \sum_{(u, v)} (1.0 - S_{comp}(u, v))$$
   *Lower $g(n)$ means the materials along the path are compositionally very close.*
2. **Heuristic Function $h(n)$:** The estimated remaining deficit to satisfy the engineering goal:
   $$h(n) = \max\left(0, \frac{\text{Required\_TS} - TS(n)}{\text{Required\_TS}}\right) + \max\left(0, \frac{\text{Required\_Duct} - \text{Duct}(n)}{\text{Required\_Duct}}\right)$$
   - If material $n$ already meets both required thresholds, $h(n) = 0$.
   - If material $n$ falls short of tensile strength or ductility, $h(n) > 0$ proportional to the fractional deficit.

### 4.3 Engineering Honesty on Optimality
- Standard A* optimality requires an **admissible heuristic** ($h(n) \le h^*(n)$, never overestimating the true remaining cost).
- While our normalized property shortfall is bounded and non-negative, because real metallurgical property changes are complex and non-linear, this heuristic provides **guiding pressure** toward stronger alloys rather than an absolute proof of global graph optimality.
- Therefore, in MaterialMind AI, A* is formally documented as **heuristic-guided search**.

### 4.4 Step-by-Step Example
- Start: AISI 4140 ($TS = 814\text{ MPa}$). Target requirement: $TS \ge 1000\text{ MPa}$.
- Start node $n_0$: $g(n_0) = 0.0$. Shortfall $h(n_0) = \frac{1000 - 814}{1000} = 0.186$. Total $f(n_0) = 0.186$.
- Pop $n_0$; inspect neighbors:
  - Neighbor A (AISI 4130, $TS=670\text{ MPa}$, edge cost $0.05$):  
    $g(A) = 0.05$, $h(A) = \frac{1000-670}{1000} = 0.330 \implies f(A) = 0.380$.
  - Neighbor B (AISI 4140 Q&T 25mm, $TS=1020\text{ MPa}$, edge cost $0.02$):  
    $g(B) = 0.02$, $h(B) = 0.000$ (Goal met!) $\implies f(B) = 0.020$.
- Priority queue pops Neighbor B first ($f=0.020 < 0.380$). Goal condition verified!
- Path returned: `[AISI 4140 (Normalized) -> AISI 4140 (Q&T)]`, total cost: $0.02$.

### 4.5 Complexity & Project Role
- **Time Complexity:** $O(E \log V)$ with binary heap (`heapq`).
- **Space Complexity:** $O(V)$ to store priority queue and path pointers.
- **Implementation:** [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py)

---

# 5. Local Search (Hill-Climbing Optimization)

### 5.1 Purpose
Takes an initial feasible substitute and explores immediate neighboring alloys to find a localized candidate that achieves a superior multi-criteria balance across property suitability, composition closeness, ductility reserve, and cost.

### 5.2 Objective Function (Evaluation Metric)
For candidate material $M$ compared against reference material $R$:
$$\text{Score}(M) = w_{comp} \cdot S_{comp}(M, R) + w_{mech} \cdot S_{mech}(M, R) + w_{duct} \cdot \text{DuctScore}(M) - w_{cost} \cdot \text{CostPenalty}(M)$$
Default weights:
$$w_{comp} = 0.40, \quad w_{mech} = 0.35, \quad w_{duct} = 0.15, \quad w_{cost} = 0.10$$

### 5.3 Working Principle (Steepest Ascent Hill-Climbing)
1. Evaluate score of current material $M_{curr}$: $\text{Score}(M_{curr})$.
2. Examine all adjacent neighbors of $M_{curr}$ in the similarity graph.
3. Compute the score for each neighbor.
4. If the best neighbor $M_{best}$ has a higher score than $M_{curr}$:
   $$M_{curr} \leftarrow M_{best}$$
   Record transition in the optimization trajectory.
5. If no neighbor improves the score, the algorithm halts at a **local maximum**.

### 5.4 Step-by-Step Example
- Initial Candidate: Node 25 ($TS = 965\text{ MPa}$, Score = $0.7198$).
- Neighbors evaluated: {Node 20 (Score = $0.8043$), Node 21 (Score = $0.6950$), Node 26 (Score = $0.7100$)}.
- Step 1: Climbs to Node 20 because its score ($0.8043$) is significantly higher.
- Step 2: Evaluates all neighbors of Node 20. None exceed $0.8043$.
- Halts at Step 2; returns Node 20 as the locally optimized recommendation.

### 5.5 Complexity & Project Role
- **Time Complexity:** $O(k \cdot \text{deg})$ where $k$ is the number of climbing steps (typically $2 - 5$) and $\text{deg}$ is the average node degree ($\le 8$).
- **Space Complexity:** $O(1)$ auxiliary space beyond storing the trajectory path.
- **Implementation:** [`src/search_algorithms.py`](file:///d:/Antigravity/Aiml_Project/src/search_algorithms.py)

---

# 6. Multi-Criteria Ranking Methodology

After hard constraints eliminate invalid materials, remaining feasible candidates are ranked deterministically:

| Factor | Weight | Formula / Basis |
|---|---|---|
| **Composition Similarity ($S_{comp}$)** | **35%** | $\frac{1}{1 + \|\mathbf{x}_{comp} - \mathbf{x}_{ref}\|}$ |
| **Mechanical Similarity ($S_{mech}$)** | **35%** | $\frac{1}{1 + \text{mean}(|\Delta P| / P_{max})}$ across TS, YS, Ductility |
| **Ductility Reserve Margin** | **15%** | $\min\left(1.0, \frac{\text{Ductility}}{30\%}\right)$ |
| **Cost Advantage** | **15%** | $\frac{\text{Price}_{ref} - \text{Price}_{cand}}{\text{Price}_{ref}}$ (clamped in $[-1.0, +1.0]$, 0 if unknown) |

$$\text{Final Suitability Score} = 0.35 \cdot S_{comp} + 0.35 \cdot S_{mech} + 0.15 \cdot \text{DuctMargin} + 0.15 \cdot \text{CostAdvantage}$$

Candidates are sorted in descending order of this score, ensuring 100% deterministic ranking.
