# Customer Personality Analysis — Unsupervised Machine Learning

A complete unsupervised machine learning pipeline that segments customers from a retail marketing dataset into meaningful clusters, enabling targeted marketing strategies based on spending habits, demographics, and lifestyle.

---

## Project Overview

This project applies unsupervised learning techniques to a real-world marketing dataset to uncover hidden patterns in customer behaviour. By combining **dimensionality reduction** (PCA) with **clustering** (Agglomerative Clustering), the pipeline transforms raw customer data into actionable customer segments — each with a distinct profile in terms of income, spending, family structure, and promotional engagement.

The end goal is to help a business answer: *"Who are our customers, and how should we approach each group differently?"*

---

## Dataset

**File:** `marketing_campaign.csv` (tab-separated)

The dataset contains demographic information, purchasing behaviour, and campaign response history for over 2,000 customers. Key original features include:

| Category | Features |
|---|---|
| Demographics | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| Purchasing | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| Engagement | `Recency`, `NumDealsPurchases`, `Dt_Customer` |
| Campaigns | `AcceptedCmp1` through `AcceptedCmp5`, `Response`, `Complain` |

---

## Tech Stack & Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
yellowbrick
mpl_toolkits (3D plotting)
```

Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn yellowbrick
```

---

## Pipeline Walkthrough

### 1. Data Loading & Initial Inspection

The dataset is loaded from a tab-separated CSV. An initial inspection using `.info()` and `.head()` reveals the data types, shape, and presence of missing values.

```python
data = pd.read_csv('marketing_campaign.csv', sep="\t")
```

---

### 2. Data Cleaning

- **Missing values** are dropped entirely to ensure model integrity.
- **Outliers** are removed by capping:
  - `Age < 90` — removes implausible birthdates
  - `Income < 600,000` — removes extreme income outliers that would skew scaling

After cleaning, the dataset retains a high-quality subset of customers.

---

### 3. Feature Engineering

Several new features are derived to better capture customer reality:

| New Feature | Description |
|---|---|
| `Age` | Calculated as `2021 - Year_Birth` |
| `Spent` | Total spend across all product categories (Wines + Fruits + Meat + Fish + Sweets + Gold) |
| `Living_With` | Simplified marital status: `"Partner"` or `"Alone"` |
| `Children` | Sum of `Kidhome` and `Teenhome` |
| `Family_Size` | Household size derived from living situation and number of children |
| `Is_Parent` | Binary flag: `1` if the customer has any children, `0` otherwise |
| `Customer_For` | Number of days the customer has been enrolled, relative to the most recent join date |

Education levels are consolidated into three tiers:
- `Undergraduate` (Basic, 2n Cycle)
- `Graduate` (Graduation)
- `Postgraduate` (Master, PhD)

Redundant columns (`Marital_Status`, `Dt_Customer`, `Year_Birth`, `ID`, `Z_CostContact`, `Z_Revenue`) are dropped.

---

### 4. Encoding & Scaling

- **Label Encoding** is applied to remaining categorical features (`Education`, `Living_With`) to convert them to numeric form.
- **Standard Scaling** (`StandardScaler`) normalises all features to zero mean and unit variance, ensuring no single feature dominates the clustering due to scale differences.
- Campaign-response columns (`AcceptedCmp1–5`, `Complain`, `Response`) are excluded from the clustering feature set to keep the model focused on inherent customer characteristics rather than past behaviour.

---

### 5. Dimensionality Reduction — PCA

**Principal Component Analysis (PCA)** reduces the high-dimensional feature space down to **3 principal components**. This serves two purposes:

1. Reduces noise and redundancy in the data
2. Allows the clusters to be visualised in 3D space

```python
pca = PCA(n_components=3)
PCA_ds = pd.DataFrame(pca.transform(scaled_ds), columns=["col1", "col2", "col3"])
```

The resulting 3D scatter plot reveals natural groupings in the data before clustering is applied.

---

### 6. Determining Optimal Clusters — Elbow Method

The **Elbow Method** via `yellowbrick`'s `KElbowVisualizer` is used to identify the optimal number of clusters for KMeans, providing a principled data-driven justification for the cluster count chosen.

```python
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
```

The elbow curve identifies **4 clusters** as the optimal choice.

---

### 7. Clustering — Agglomerative Clustering

**Agglomerative (Hierarchical) Clustering** is used as the final clustering algorithm, with `n_clusters=4`. This bottom-up approach progressively merges data points and is well-suited to customer segmentation tasks where cluster shapes may not be perfectly spherical.

```python
AC = AgglomerativeClustering(n_clusters=4)
yhat_AC = AC.fit_predict(PCA_ds)
```

Cluster labels are assigned back to both the PCA-reduced dataframe and the original dataframe for analysis.

---

### 8. Cluster Analysis & Visualisation

The four clusters are profiled across multiple dimensions with rich visualisations:

#### Distribution
- **Countplot** showing the size of each cluster

#### Income vs. Spending
- **Scatter plot** revealing the income-to-spend relationship per cluster — a key axis for segmenting high-value vs. budget customers

#### Spending Distribution
- **Swarm + Boxen plot** showing the spread and median of total spending per cluster

#### Promotional Engagement
- **Countplot** of accepted promotions by cluster — revealing which segments are promotion-driven
- **Boxen plot** of number of deals purchased per cluster

#### Personal & Demographic Profiles
- **KDE Joint plots** of spending vs. each personal feature (`Age`, `Children`, `Family_Size`, `Education`, etc.) coloured by cluster — providing a rich multidimensional portrait of each segment

---

## Results — Customer Segments

The model identifies **4 distinct customer clusters**, each with a clear demographic profile and corresponding marketing strategy.

---

### Cluster 0 — Busy Family Parents

**Profile:**
- Definitely a parent; most have a **teenager at home**
- Family size of **2–4 members**, including single-parent households
- Relatively **older** age group

**Marketing Strategy — Convenience & Family Value:**
- Offer **multi-buy discounts** on household staples and non-perishables (e.g. "Buy 2, Get 1 Free" on cereals or cleaning supplies)
- Promote **ready-made meals and meal kits** sized for 3–4 people
- Use **traditional marketing channels** (flyers, newspaper inserts) alongside digital ads given the older demographic

---

### Cluster 1 — Young Parents with Small Children

**Profile:**
- Majority are parents, typically with **one young child** (not teenagers)
- Family size of **up to 3 members**
- Relatively **younger** age group

**Marketing Strategy — Child-Centric & Premium Quality:**
- Highlight **fresh produce, organic, and health-focused baby/toddler items**
- Run targeted **social media campaigns** and use app-based **personalised coupons** on baby/toddler brands
- Create **in-store family-friendly experiences** (e.g. kids' samples, family parking)

---

### Cluster 2 — High-Income Couples & Individuals

**Profile:**
- Definitely **not a parent**
- Family size of **up to 2 members**; slight majority are couples over singles
- **Spans all age groups**
- **High-income** group with strong spending power

**Marketing Strategy — Premium & Gourmet Focus:**
- Promote **high-end meats, imported cheeses, fine wines, and specialty/international foods**
- Emphasise **quality and unique experiences** over discounts
- Create **premium bundle kits** (e.g. "Gourmet Pasta Night") and use **magazine-style email newsletters** to feature exclusive new products

---

### Cluster 3 — Larger, Budget-Conscious Families

**Profile:**
- Definitely a parent; majority have a **teenager at home**
- Largest family sizes — **2–5 members**
- Relatively **older** age group
- **Lower-income** group

**Marketing Strategy — Savings & Volume Discounts:**
- Prioritise **store-brand/private-label products** across all categories for better value
- Promote **bulk buying options** and large family-size products
- Heavily publicise **weekly circular sales** and implement a **loyalty programme** with high-value points or cash-off rewards to drive repeat visits

---

## Project Structure

```
├── Machine_Learning.ipynb   # Main notebook — full pipeline
└── marketing_campaign.csv   # Source dataset (tab-separated)
```

---

## Key ML Concepts Used

- **Unsupervised Learning** — no labels are provided; the model discovers structure from the data itself
- **Principal Component Analysis (PCA)** — linear dimensionality reduction to capture maximum variance in fewer dimensions
- **Agglomerative Clustering** — hierarchical bottom-up clustering algorithm
- **Elbow Method** — heuristic for selecting the optimal number of clusters
- **Feature Engineering** — domain-driven construction of meaningful features from raw data
- **Standard Scaling** — normalisation prerequisite for distance-based algorithms

---

## Acknowledgements

Dataset sourced from a public marketing campaign dataset commonly used for customer personality analysis tasks in the data science community.
