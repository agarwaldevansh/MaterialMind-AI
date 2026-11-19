# MaterialMind AI — Machine Learning Model & Algorithm Explanations

**Academic Level:** Third-Year Computer Engineering / Data Science  
**Target Audience:** Engineering Students, Faculty Reviewers, and Evaluators  
**Scope:** Conceptual, Mathematical, and Practical Explanation of All Machine Learning Algorithms Implemented in Step 2.

---

## Table of Contents
1. [Core Regression Algorithms](#1-core-regression-algorithms)
   - [1.1 Multiple Linear Regression](#11-multiple-linear-regression)
   - [1.2 Polynomial Regression](#12-polynomial-regression)
   - [1.3 Decision Tree Regressor](#13-decision-tree-regressor)
   - [1.4 Gradient Descent Optimization (SGDRegressor)](#14-gradient-descent-optimization-sgdregressor)
2. [Classification Algorithms (Strength Tiers)](#2-classification-algorithms-strength-tiers)
   - [2.1 Decision Tree Classifier](#21-decision-tree-classifier)
   - [2.2 K-Nearest Neighbors (KNN)](#22-k-nearest-neighbors-knn)
   - [2.3 Support Vector Machine (SVM)](#23-support-vector-machine-svm)
   - [2.4 Gaussian Naive Bayes](#24-gaussian-naive-bayes)
3. [Ensemble Learning Techniques](#3-ensemble-learning-techniques)
   - [3.1 Voting Classifier (Soft Voting)](#31-voting-classifier-soft-voting)
   - [3.2 AdaBoost Classifier (Adaptive Boosting)](#32-adaboost-classifier-adaptive-boosting)
   - [3.3 Stacking Classifier (Stacked Generalization)](#33-stacking-classifier-stacked-generalization)
4. [Unsupervised Clustering Algorithms](#4-unsupervised-clustering-algorithms)
   - [4.1 K-Means Clustering](#41-k-means-clustering)
   - [4.2 Agglomerative Hierarchical Clustering](#42-agglomerative-hierarchical-clustering)
5. [Dimensionality Reduction](#5-dimensionality-reduction)
   - [5.1 Principal Component Analysis (PCA)](#51-principal-component-analysis-pca)
6. [Strict Anti-Leakage & Group Splitting Strategy](#6-strict-anti-leakage--group-splitting-strategy)

---

# 1. Core Regression Algorithms

---

### 1.1 Multiple Linear Regression

- **What it does:**  
  Models a continuous relationship between multiple input features (chemical elements and temperature) and a continuous mechanical property (e.g. Tensile Strength) as a linear combination.

- **Why we need it:**  
  Serves as the foundational baseline algorithm in predictive modeling. In metallurgy, empirical equations like Pickering’s or Gladman’s equations are linear formulas relating carbon, manganese, and grain size to yield strength.

- **How it works in simple language:**  
  It finds a hyperplane that minimizes the sum of squared vertical differences (residuals) between actual measured strength and predicted strength.

- **Mathematical Formula:**  
  $$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p$$
  Where:
  - $\hat{y}$: Predicted mechanical property (e.g., Tensile Strength in MPa).
  - $\beta_0$: Intercept (base strength of pure annealed unalloyed iron).
  - $\beta_j$: Regression coefficient representing the change in strength per $1\text{ wt}\%$ increase in element $j$.
  - $x_j$: Weight percent of element $j$ (e.g., Carbon, Manganese).

- **Input Features:** Standardized composition features ($C, Mn, Si, Cr, Ni, Mo, \dots$).
- **Target:** Continuous mechanical properties (`tensile_strength_mpa`, `yield_strength_mpa`, `ductility_pct`, `proof_stress_mpa`).
- **Output:** Continuous predicted value in engineering units (MPa or %).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$).
- **Simple Example:**  
  Suppose baseline pure iron has $\beta_0 = 300\text{ MPa}$. Carbon adds $\beta_C = 800\text{ MPa per 1\% C}$ and Manganese adds $\beta_{Mn} = 150\text{ MPa per 1\% Mn}$.  
  For steel with $0.40\%\text{ C}$ and $0.80\%\text{ Mn}$:  
  $$\hat{y} = 300 + (800 \times 0.40) + (150 \times 0.80) = 300 + 320 + 120 = 740\text{ MPa}$$
- **Implementation File:** [`src/regression_models.py`](file:///d:/Antigravity/Aiml_Project/src/regression_models.py)

---

### 1.2 Polynomial Regression

- **What it does:**  
  Extends linear regression by adding cross-product interaction terms (e.g., $x_1 \times x_2$) to capture nonlinear synergies between alloying elements without switching to a black-box model.

- **Why we need it:**  
  In steel metallurgy, elements do not act in isolation. For example, Chromium and Molybdenum act synergistically ($Cr \times Mo$) to form complex carbides and enhance hardenability far more than either element added alone.

- **How it works in simple language:**  
  It creates new synthetic features by multiplying pairs of existing features together. It then fits a linear model on this expanded feature space, stabilized with Ridge ($L_2$) regularization to prevent severe collinear overfitting on unseen steel grades.

- **Mathematical Formula:**  
  $$\hat{y} = \beta_0 + \sum_{i=1}^p \beta_i x_i + \sum_{i=1}^p \sum_{j=i+1}^p \beta_{ij} (x_i \cdot x_j)$$
  Objective function with Ridge regularization:  
  $$\min_{\beta} \left\{ \sum_{k=1}^N (y_k - \hat{y}_k)^2 + \alpha \sum \beta^2 \right\}$$

- **Input Features:** Normalized composition elements and interaction pairs.
- **Target:** Mechanical properties (e.g., `tensile_strength_mpa`, `proof_stress_mpa`).
- **Output:** Continuous predicted mechanical property.
- **Evaluation Metrics:** MAE, MSE, RMSE, $R^2$.
- **Simple Example:**  
  Adding $1\%\text{ Cr}$ alone adds $50\text{ MPa}$. Adding $0.5\%\text{ Mo}$ alone adds $60\text{ MPa}$. But when both are present, secondary hardening precipitates $M_{23}C_6$ carbides, contributing an interaction boost $\beta_{Cr \cdot Mo} (x_{Cr} \cdot x_{Mo}) = 40 \times (1.0 \times 0.5) = +20\text{ MPa}$.
- **Implementation File:** [`src/regression_models.py`](file:///d:/Antigravity/Aiml_Project/src/regression_models.py)

---

### 1.3 Decision Tree Regressor

- **What it does:**  
  Predicts mechanical properties by learning a sequence of simple binary decision rules (if-then splits) directly from metallurgical features.

- **Why we need it:**  
  Steel microstructures and properties change discontinuously across metallurgical regimes (e.g., above $0.30\%\text{ C}$, steel can form martensite upon quenching; below $0.20\%\text{ C}$, it remains mostly ferrite-pearlite). Trees naturally handle these discrete regime shifts.

- **How it works in simple language:**  
  The algorithm scans all input features and split points to find the partition that minimizes the variance (MSE) of target values in the resulting two subsets. A new material travels down the tree following decisions until it reaches a leaf node, which predicts the average strength of historical materials in that leaf.

- **Mathematical Splitting Criterion:**  
  $$\text{MSE}(S) = \frac{1}{|S|} \sum_{i \in S} (y_i - \bar{y}_S)^2$$
  $$\Delta \text{MSE} = \text{MSE}(S) - \left( \frac{|S_L|}{|S|} \text{MSE}(S_L) + \frac{|S_R|}{|S|} \text{MSE}(S_R) \right)$$

- **Input Features:** Composition elements, ratios, and processing indicators.
- **Target:** `tensile_strength_mpa`, `yield_strength_mpa`, `ductility_pct`.
- **Output:** Predicted value (mean target value of leaf node).
- **Evaluation Metrics:** MAE, RMSE, $R^2$.
- **Simple Example:**  
  Root node asks: *Is Carbon > 0.35%?*  
  - If Yes: *Is Cr + Mo > 1.0%?* -> Leaf predicts $1050\text{ MPa}$.  
  - If No: *Is Carbon < 0.15%?* -> Leaf predicts $420\text{ MPa}$.
- **Implementation File:** [`src/regression_models.py`](file:///d:/Antigravity/Aiml_Project/src/regression_models.py)

---

### 1.4 Gradient Descent Optimization (SGDRegressor)

- **What it does:**  
  Iteratively estimates the optimal linear regression weights using Stochastic Gradient Descent (SGD) rather than analytical closed-form matrix inversion (Normal Equation).

- **Why we need it:**  
  To understand how large-scale machine learning models (and deep neural networks) optimize parameters when datasets are too massive to compute matrix inverses $(X^T X)^{-1}$.

- **Important Distinction:**  
  Gradient descent is an **optimization algorithm**, not a distinct model family. It is an iterative method for solving the parameters of a Linear Regression model.

- **How it works in simple language:**  
  Imagine standing on a foggy hillside trying to reach the lowest valley (minimum error). You look at the slope beneath your feet and take a step in the direction of steepest descent. You repeat this step-by-step until the slope becomes nearly flat.

- **Mathematical Update Rule:**  
  $$\beta_j \leftarrow \beta_j - \eta \cdot \frac{\partial \mathcal{L}}{\partial \beta_j}$$
  For mean squared error loss:  
  $$\frac{\partial \mathcal{L}}{\partial \beta_j} = -2 (y_i - \hat{y}_i) x_{ij}$$
  Where $\eta$ is the learning rate.

- **Input Features:** Standardized composition features (standardization is mandatory so gradients do not oscillate wildly).
- **Target:** Mechanical properties.
- **Output:** Continuous predicted value.
- **Evaluation Metrics:** MAE, RMSE, $R^2$.
- **Implementation File:** [`src/regression_models.py`](file:///d:/Antigravity/Aiml_Project/src/regression_models.py)

---

# 2. Classification Algorithms (Strength Tiers)

---

### 2.1 Decision Tree Classifier

- **What it does:**  
  Classifies steels into one of three strength categories: Low Strength (<600 MPa), Medium Strength (600–950 MPa), or High Strength (≥950 MPa) using hierarchical if-then rules.

- **Why we need it:**  
  Gives engineering practitioners direct transparency into the decision thresholds (e.g. "If carbon > 0.38% and Cr+Mo > 1.1%, predict High Strength").

- **How it works in simple language:**  
  At each node, it selects the feature and threshold that maximizes purity (minimizes Gini Impurity) of the split child nodes.

- **Mathematical Criterion (Gini Impurity):**  
  $$I_G(S) = 1 - \sum_{k=1}^C p_k^2$$
  Where $p_k$ is the fraction of materials in node $S$ belonging to class $k$. A pure node (containing only High Strength steels) has $I_G = 0$.

- **Input Features:** Composition elements and ratios (strictly excluding tensile strength and yield strength).
- **Target:** `strength_category` ('Low Strength', 'Medium Strength', 'High Strength').
- **Output:** Discrete class label and probability vector.
- **Evaluation Metrics:** Accuracy, Macro Precision, Macro Recall, Macro F1-score, Confusion Matrix.
- **Implementation File:** [`src/classification_models.py`](file:///d:/Antigravity/Aiml_Project/src/classification_models.py)

---

### 2.2 K-Nearest Neighbors (KNN)

- **What it does:**  
  Classifies an unknown steel grade based on the majority category among its $k$ closest neighbors in chemical composition space.

- **Why we need it:**  
  Emulates how metallurgists traditionally select alternative materials: finding historical alloy grades that possess the most similar chemical formulations.

- **How it works in simple language:**  
  All alloy compositions are plotted in multi-dimensional space (scaled with StandardScaler so $1\%\text{ C}$ does not get overwhelmed by $98\%\text{ Fe}$). When a new alloy arrives, the algorithm measures Euclidean distance to all training materials, picks the 5 closest alloys, and votes on the strength category.

- **Mathematical Distance Metric:**  
  $$d(\mathbf{x}_a, \mathbf{x}_b) = \sqrt{\sum_{j=1}^p (x_{a,j} - x_{b,j})^2}$$

- **Input Features:** Standardized composition features ($C, Mn, Si, Cr, \dots$).
- **Target:** `strength_category`.
- **Output:** Class label with distance-weighted probability.
- **Evaluation Metrics:** Macro F1, Accuracy, Confusion Matrix.
- **Implementation File:** [`src/classification_models.py`](file:///d:/Antigravity/Aiml_Project/src/classification_models.py)

---

### 2.3 Support Vector Machine (SVM)

- **What it does:**  
  Finds the optimal separating hyperplane that maximizes the margin (geometric gap) between different material strength classes.

- **Why we need it:**  
  Highly effective in multi-dimensional feature spaces where nonlinear boundaries separate low-alloy grades from high-strength tool and maraging steels.

- **How it works in simple language:**  
  If two classes cannot be separated by a straight line in raw composition space, the Radial Basis Function (RBF) kernel implicitly projects features into an infinite-dimensional space where a clean planar separation is possible.

- **Mathematical Kernel Formula (RBF Kernel):**  
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2)$$
  Where $\gamma = \frac{1}{p \cdot \text{Var}(X)}$ dictates the influence radius of individual support vectors.

- **Input Features:** Standardized chemical composition features.
- **Target:** `strength_category`.
- **Output:** Predicted class label.
- **Evaluation Metrics:** Macro F1, Accuracy, Confusion Matrix.
- **Implementation File:** [`src/classification_models.py`](file:///d:/Antigravity/Aiml_Project/src/classification_models.py)

---

### 2.4 Gaussian Naive Bayes

- **What it does:**  
  Calculates the posterior probability of each strength class using Bayes' Theorem under the assumption that all alloying elements are conditionally independent given the class.

- **Why we need it:**  
  Provides a fast, probabilistic benchmark model. It demonstrates the consequences when the independence assumption is violated (as elements like C, Mn, and Cr frequently correlate).

- **How it works in simple language:**  
  It fits a normal (Gaussian) bell curve to the distribution of each element within each strength category. To classify a new steel, it multiplies the probabilities of observing each element's value and chooses the class with the highest overall likelihood.

- **Mathematical Formula:**  
  $$P(C_k \mid \mathbf{x}) \propto P(C_k) \prod_{j=1}^p \frac{1}{\sqrt{2\pi \sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)$$

- **Input Features:** Standardized composition features.
- **Target:** `strength_category`.
- **Output:** Probability per class; predicted class with highest posterior.
- **Evaluation Metrics:** Accuracy, Macro F1, Recall.
- **Implementation File:** [`src/classification_models.py`](file:///d:/Antigravity/Aiml_Project/src/classification_models.py)

---

# 3. Ensemble Learning Techniques

---

### 3.1 Voting Classifier (Soft Voting)

- **What it does:**  
  Combines the predicted class probabilities from multiple diverse base models (Decision Tree, KNN, SVM, Gaussian NB) and selects the class with the highest average probability.

- **Why we need it:**  
  Different models make errors in different regions of the feature space. Averaging probabilities smooths out individual model noise and reduces variance.

- **Mathematical Decision Rule:**  
  $$\hat{y} = \arg\max_{k} \frac{1}{M} \sum_{m=1}^M P_m(y = C_k \mid \mathbf{x})$$

- **Implementation File:** [`src/ensemble_models.py`](file:///d:/Antigravity/Aiml_Project/src/ensemble_models.py)

---

### 3.2 AdaBoost Classifier (Adaptive Boosting)

- **What it does:**  
  Trains a sequence of 50 shallow decision trees ("stumps"). In each iteration, sample weights are increased for previously misclassified steels so subsequent trees focus on correcting hard-to-classify materials.

- **Why we need it:**  
  Converts a collection of weak learners into a strong, highly accurate classifier capable of resolving difficult boundary cases between Medium and High strength steels.

- **How it works in simple language:**  
  Think of a student preparing for an exam: after taking a practice test, the student identifies questions they got wrong and spends the next study session focusing specifically on those difficult questions.

- **Mathematical Sample Weight Update:**  
  $$w_{i}^{(t+1)} = w_{i}^{(t)} \exp\left( \alpha_t \cdot \mathbb{I}(y_i \neq h_t(\mathbf{x}_i)) \right)$$
  Where $\alpha_t = \frac{1}{2} \ln \left( \frac{1 - \epsilon_t}{\epsilon_t} \right)$ is the weight assigned to tree $t$ based on its error rate $\epsilon_t$.

- **Result in MaterialMind AI:** Achieved **67.2% test accuracy** and **0.693 macro F1-score**, the highest of all classification models.
- **Implementation File:** [`src/ensemble_models.py`](file:///d:/Antigravity/Aiml_Project/src/ensemble_models.py)

---

### 3.3 Stacking Classifier (Stacked Generalization)

- **What it does:**  
  Uses the out-of-fold predictions from multiple base models (DT, KNN, SVM, GNB) as inputs to train a higher-level meta-learner (Logistic Regression).

- **Why we need it:**  
  Instead of assigning equal weight to every model (as in simple voting), Stacking *learns* which model is most reliable for different compositions.

- **How it prevents data leakage:**  
  Out-of-fold cross-validation ($cv=5$) ensures the meta-model is trained on predictions for data the base models did not see during fitting.

- **Implementation File:** [`src/ensemble_models.py`](file:///d:/Antigravity/Aiml_Project/src/ensemble_models.py)

---

# 4. Unsupervised Clustering Algorithms

---

### 4.1 K-Means Clustering

- **What it does:**  
  Partitions steel materials into $k$ non-overlapping clusters based on Euclidean proximity in standardized chemical composition space.

- **Why we need it:**  
  Identifies natural metallurgical families (e.g. low-carbon structural steels, Cr-Mo alloy steels, Ni-Cr-Mo steels, high-alloy specialty steels) without requiring human labels.

- **How it works in simple language:**  
  1. Initialize $k$ random cluster centers (centroids) using $k\text{-means++}$.
  2. Assign each alloy to its nearest centroid.
  3. Recompute the centroid as the mean of all alloys assigned to that cluster.
  4. Repeat until centroid coordinates stop changing.

- **Mathematical Objective (Inertia):**  
  $$J = \sum_{j=1}^k \sum_{\mathbf{x}_i \in S_j} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2$$

- **Selecting Optimal $k$:**  
  - **Elbow Method:** Plot Inertia vs $k$. The curve drops rapidly then flattens, forming an "elbow" around $k=4$.
  - **Silhouette Score:** Evaluates how close each sample is to its assigned cluster relative to neighboring clusters:
    $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
    At $k=4$, K-Means achieved an outstanding Silhouette Score of **0.812**, confirming clean cluster separation.

- **Implementation File:** [`src/clustering_models.py`](file:///d:/Antigravity/Aiml_Project/src/clustering_models.py)

---

### 4.2 Agglomerative Hierarchical Clustering

- **What it does:**  
  Builds a bottom-up hierarchical tree (dendrogram) of material clusters by successively merging the two most similar alloy groups.

- **Why we need it:**  
  Shows the nested hierarchy of alloy families (e.g. AISI 4130 and AISI 4140 merge first, then merge into the broader Cr-Mo family, which subsequently merges with Ni-Cr-Mo steels).

- **How it works in simple language:**  
  Every material starts as its own individual cluster. The algorithm iteratively joins the two clusters whose merger causes the smallest increase in total within-cluster variance (Ward's method), until all materials form one tree.

- **Mathematical Linkage Criterion (Ward's Minimum Variance):**  
  $$\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B} \|\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\|^2$$

- **Comparison with K-Means:**  
  Achieved an **Adjusted Rand Index (ARI) of 0.862**, proving strong mathematical agreement between K-Means and Hierarchical Clustering.
- **Implementation File:** [`src/clustering_models.py`](file:///d:/Antigravity/Aiml_Project/src/clustering_models.py)

---

# 5. Dimensionality Reduction

---

### 5.1 Principal Component Analysis (PCA)

- **What it does:**  
  Transforms 23 correlated composition features into orthogonal (uncorrelated) linear axes called Principal Components, ordered by the amount of variance they explain.

- **Why we need it:**  
  Humans cannot visualize 23-dimensional chemistry space. PCA reduces this space to 2 principal components while preserving the vast majority of metallurgical variance.

- **How it works in simple language:**  
  Imagine projecting the 3D shadow of a flying bird onto a 2D wall. If you position the light correctly, the 2D shadow captures the bird's full shape with minimal loss of detail. PCA finds the exact angle that preserves the widest spread of data points.

- **Mathematical Eigenvalue Decomposition:**  
  1. Standardize data matrix $X$.
  2. Compute covariance matrix $\Sigma = \frac{1}{N-1} X^T X$.
  3. Solve eigenvalue equation $\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$.
  4. The eigenvector $\mathbf{v}_1$ with largest eigenvalue $\lambda_1$ defines PC1.

- **Variance Captured in MaterialMind AI:**  
  - **PC1:** Explains **32.63%** of total variance (dominated by Fe vs Total Alloying).
  - **PC2:** Explains **13.95%** of total variance (dominated by Cr, Mo, and Ni additions).
  - **Top 10 Components:** Capture **86.02%** of all metallurgical variance.
- **Implementation File:** [`src/dimensionality_reduction.py`](file:///d:/Antigravity/Aiml_Project/src/dimensionality_reduction.py)

---

# 6. Strict Anti-Leakage & Group Splitting Strategy

### A. GroupShuffleSplit Methodology
- In **Dataset 1**, multiple test rows share the same steel grade `name` (evaluated across bar diameters: 13mm, 25mm, 50mm, 100mm).
- Using random train/test splitting would place the 25mm test in train and the 50mm test in test, leading to **data leakage** and artificially inflated $R^2$ scores.
- **Our Solution:** `GroupShuffleSplit(groups=df['name'])` ensures all instances of a steel grade are exclusively in train (182 grades) OR test (46 grades).
- In **Dataset 2**, `GroupShuffleSplit(groups=df['alloy_code'])` keeps all temperature measurements (27°C–650°C) of each alloy strictly together in either train or test.

### B. Target Leakage Prevention Guarantee
- Feature pipelines use Scikit-learn `Pipeline([('scaler', StandardScaler()), ...])`, fitting mean and variance **strictly on training partitions**.
- Targets (`tensile_strength_mpa`, `yield_strength_mpa`, `ductility_pct`, `proof_stress_mpa`) are never accessible to input feature matrices.
- The `yield_tensile_ratio_DISPLAY_ONLY` feature is strictly quarantined for post-inference presentation.
