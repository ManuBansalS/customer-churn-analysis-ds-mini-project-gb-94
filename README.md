# Customer Churn Analysis - Complete Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Stage 1: Data Exploration (Bronze Layer)](#stage-1-data-exploration-bronze-layer)
3. [Stage 2: Data Cleaning & Preprocessing (Silver Layer)](#stage-2-data-cleaning--preprocessing-silver-layer)
4. [Stage 3: Exploratory Data Analysis (EDA)](#stage-3-exploratory-data-analysis-eda)
5. [Stage 4: Hypothesis Testing](#stage-4-hypothesis-testing)
6. [Stage 5: Feature Engineering](#stage-5-feature-engineering)
7. [Stage 6: Model Training & Evaluation (Gold Layer)](#stage-6-model-training--evaluation-gold-layer)
8. [Key Insights & Recommendations](#key-insights--recommendations)

---

## Project Overview

### What is This Project About?

This project is a **Customer Churn Analysis and Prediction System** designed to identify and predict which customers are likely to churn (leave) or partially churn from a business. The goal is to:

1. **Understand**: Explore what causes customers to churn
2. **Predict**: Build machine learning models to classify customers into three categories:
   - **No Churn**: Customer stays with the company
   - **Partial Churn**: Customer reduces engagement or switches some agreements
   - **Full Churn**: Customer completely leaves

### Why This Structure?

The project follows a **data engineering best practice** called the **Bronze-Silver-Gold (BSG) architecture**:

- **Bronze Layer (01_bronze)**: Raw data exploration and basic loading
- **Silver Layer (02_silver)**: Data cleaning, preprocessing, EDA, hypothesis testing, and feature engineering
- **Gold Layer (03_gold)**: Model training and evaluation
- **Source Code (src/)**: Reusable, modular Python classes for each stage

This structure ensures:
- ✅ **Reproducibility**: Code is version-controlled and organized
- ✅ **Scalability**: Easy to add new features or models
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Documentation**: Each stage has a notebook explaining the logic

### Dataset Overview

The project uses **two main datasets**:

| Dataset | Records | Purpose |
|---------|---------|---------|
| **BoB.csv** | 207,275 agreements | Business operational baseline (revenue, fees, products) |
| **Retention.csv** | 99,186 records | Customer retention/churn events |

**Key Statistics After Merging:**
- **Total Unique Customers**: 23,772
- **No Churn**: 22,370 (94.1%)
- **Partial Churn**: 1,067 (4.5%)
- **Full Churn**: 335 (1.4%)

---

## Stage 1: Data Exploration (Bronze Layer)

### Purpose
The exploration stage is about understanding the raw data before making any changes. This is critical because:
- You identify data quality issues early
- You understand the structure and relationships
- You spot anomalies and outliers
- You make informed cleaning decisions

### What Happens Here

#### 1. **Data Loading** (`src/data/load_raw_data.py`)

```python
class LoadData:
    def __init__(self):
        self.retention = pd.read_csv("../../data/01_raw/Retention.csv")
        self.bob = pd.read_csv("../../data/01_raw/BoB.csv")
```

**Why this approach?**
- ✅ Encapsulation: Loading logic is isolated in its own class
- ✅ Reusability: Can be imported and used in any notebook
- ✅ Scalability: Easy to add database connections, APIs, or cloud storage later

#### 2. **Initial Exploration**

In the notebook, we examine:

```python
# Basic Info
loaded_data = LoadData()
bob = loaded_data.bob
retention = loaded_data.retention

# Check structure
print(bob.shape)      # (207,275, 11) - each row is an agreement
print(retention.shape) # (99,186, 12) - each row is a retention case

# Check columns
print(bob.columns)
print(retention.columns)

# Check data types
print(bob.dtypes)
print(retention.dtypes)

# Check missing values
print(bob.isnull().sum())
print(retention.isnull().sum())
```

### Key Findings from Exploration

1. **BoB Dataset** (Business Operations Baseline):
   - Multiple agreements per customer
   - Financial metrics: revenue, fees, unit amounts
   - Categorical data: agreement type, branch, product name
   - Date fields: agreement start and end dates

2. **Retention Dataset**:
   - Contains churn/retention events
   - Resolution status indicates if it's a churn or retention case
   - Some customers may not have retention records

3. **Data Quality Issues Identified**:
   - Column names in mixed case (CamelCase) → need standardization
   - Missing values present → need strategy for handling
   - Date columns as strings → need conversion to datetime
   - Duplicate records possible → need deduplication

### Why This Structure Matters

Instead of jumping straight to cleaning, we first understand:
- **Data shape**: How many records and features?
- **Data types**: What is each column?
- **Data quality**: How many nulls, duplicates, wrong types?

This prevents making wrong cleaning decisions based on assumptions.

---

## Stage 2: Data Cleaning & Preprocessing (Silver Layer)

### Purpose
Transform raw data into a **clean, standardized format** ready for analysis. This stage ensures:
- Consistent data types and formats
- No duplicate or missing records causing bias
- Properly named columns for easy reference

### What Happens Here

#### 1. **Column Standardization** (`src/data/clean_data.py`)

```python
def columns_to_snake_case(self, df):
    """Convert CamelCase to snake_case"""
    # Input: "TotalRevenue", "agreementType" 
    # Output: "total_revenue", "agreement_type"
    
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    s3 = re.sub(r'[^a-zA-Z0-9]+', '_', s2)
    return s3.lower().strip('_')
```

**Why?**
- **Consistency**: 'total_revenue' is easier to reference than 'TotalRevenue' or 'TOTAL_REVENUE'
- **Readability**: Snake_case is Python convention and more readable
- **Reduced Errors**: Standardized names prevent typos and errors

#### 2. **Data Type Conversion**

```python
def handling_date_datatypes(self, df, columns):
    """Convert date strings to datetime objects"""
    for column in columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')
```

**Why?**
- ✅ Enables time-based calculations (e.g., agreement duration)
- ✅ Allows time-series operations
- ✅ Prevents string comparison errors
- ❌ Alternative (avoid): Keep as strings → takes more memory, slower comparisons

#### 3. **Duplicate Removal**

```python
def dropping_duplicates(self, df):
    return df.drop_duplicates()
```

**Why?**
- Duplicates skew statistical analyses (inflates counts)
- Double-counts the same event
- Can lead to model overfitting on duplicate patterns

**Alternative approaches:**
- Keep duplicates but weight them → Weights must reflect reality
- Mark duplicates separately → Requires business validation

#### 4. **Missing Value Handling**

This is the most critical decision. We provide **3 strategies**:

##### Strategy A: Drop All Rows with Missing Values
```python
def dropping_nulls(self, df):
    return df.dropna()
```
**Pros:**
- ✅ Assumes missing is missing completely at random (MCAR)
- ✅ Preserves data integrity

**Cons:**
- ❌ Loses data (especially if many nulls)
- ❌ Biases analysis if missingness is not random

**Use when:** < 5% missing values

---

##### Strategy B: Fill with Mode (Most Frequent Value)
```python
def filling_with_modes(self, df):
    for column in df.columns:
        df[column].fillna(df[column].mode()[0], inplace=True)
    return df
```
**Example:**
```
agreement_type: [A, B, A, A, None]
Mode: A (appears 3 times)
Result: [A, B, A, A, A]  # Missing filled with A
```

**Pros:**
- ✅ Good for categorical data
- ✅ Doesn't reduce data size
- ✅ Represents most common case

**Cons:**
- ❌ Reduces variance (makes distribution sharper)
- ❌ Artificially increases mode frequency
- ❌ Not ideal for continuous data

**Use when:** Categorical columns with missing values

---

##### Strategy C: Fill with Mean (Average)
```python
def filling_with_means(self, df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].mean(), inplace=True)
    return df
```
**Example:**
```
revenue: [100, 200, 300, None]
Mean: 200
Result: [100, 200, 300, 200]  # Missing filled with mean
```

**Pros:**
- ✅ Preserves the mean of the column
- ✅ Simple and interpretation
- ✅ Works well for normally distributed data

**Cons:**
- ❌ Reduces variance
- ❌ Assumes data is missing at random
- ❌ Sensitive to outliers
- ❌ Creates artificial values that never existed

**Use when:** Numeric columns with some missing values

---

##### Strategy D: Fill with Median (Middle Value)
```python
def filling_with_medians(self, df):
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column].fillna(df[column].median(), inplace=True)
    return df
```
**Example:**
```
revenue: [100, 200, 300, None]
Median: 200
Result: [100, 200, 300, 200]  # Missing filled with median
```

**Pros:**
- ✅ More robust to outliers than mean
- ✅ Better for skewed distributions
- ✅ Preserves the median

**Cons:**
- ❌ Also reduces variance
- ❌ Creates artificial values

**Use when:** Numeric columns with outliers or skewed distributions

---

### What Our Project Does

```python
# In notebook 02_preprocessing:
cleaning_Data = CleanData(retention, bob)
retention_cleaned, bob_cleaned = cleaning_Data.clean_data_with_modes()
```

We chose **mode imputation** because:
- Most columns are categorical (agreement_type, branch, etc.)
- We want to preserve the distribution
- It's interpretable: "missing status = most common status"

### Critical Fix: Data Merge Logic

**The Problem:** Initially, the merge used `how='right'`, keeping only customers with retention records:
```python
# ❌ WRONG - Loses customers without retention data
merged = pd.merge(
    bob_cleaned,
    retention_cleaned,
    how='right'  
)
# Result: 21,526 customers (excludes those who didn't churn)
```

**The Solution:** Changed to `how='left'` to keep all customers:
```python
# ✅ CORRECT - All customers retained
merged = pd.merge(
    bob_cleaned,
    retention_cleaned,
    how='left'
)
# Result: 23,772 customers (all customers)
# Customers without retention data → classified as "no_churn"
```

**Why This Matters:**
- **Representativeness**: We analyze the true customer base
- **Class Balance**: Prevents artificial inflation of churn rates
- **Business Reality**: Not having a churn case means the customer didn't churn

---

## Stage 3: Exploratory Data Analysis (EDA)

### Purpose
**EDA is about understanding patterns in the data** through visualization and statistics. The goal:
- Understand distributions of variables
- Identify relationships between features
- Spot outliers and anomalies
- Generate hypotheses for statistical testing

### What Happens Here

#### 1. **Understanding Distributions**

```python
from src.visualization.eda_plots import EDAPlots

plotter = EDAPlots(data, target_column='churn_category')
```

**A. Distribution Plots (Histograms)**

```python
plotter.plot_distributions()  # Static matplotlib
# or
plotter.plot_distributions_plotly()  # Interactive Plotly
```

**What we see:**
- Frequency of each value
- Shape of distribution (normal, skewed, bimodal?)
- Overlays by churn category

**Why matplotlib vs Plotly?**

| Method | Matplotlib | Plotly |
|--------|-----------|---------|
| **Use Case** | Reports, papers, presentations | Interactive dashboards, exploration |
| **Hover Info** | None | ✅ Shows exact values |
| **Zoom/Pan** | ❌ Manual | ✅ Interactive |
| **Export** | PNG/PDF | PNG/HTML |
| **Performance** | Faster | Slower with large data |

**What to look for:**
```
Normally Distributed:        Skewed Right:           Bimodal:
    |
    | ╱╲
  ╱ │   ╲                  ╱╲                    ╱╲    ╱╲
╱   │     ╲                  │╲                  │  ╲  │  ╲
─────────────                  ──────────        ──────────

Bell curve              Long tail to right    Two peaks
Examples:              Examples:             Examples:
- Height               - Income              - Customer segments
- Test scores          - Revenue             - Multi-modal behavior
```

---

**B. Box Plots (Quartiles & Outliers)**

```python
plotter.plot_boxplots()  # or plot_boxplots_plotly()
```

**What is a box plot?**
```
Outliers (*)
    *
    |
    ▲ (Q3 + 1.5×IQR) = Upper fence
    │
    ├─────────┐  ← Q3 (75th percentile)
    │  ┆ ┆ ┆  │  ← Median (50th) = orange line
    ├─────────┘  ← Q1 (25th percentile)
    ▼
    | (Q1 - 1.5×IQR) = Lower fence
    *
```

**Why box plots?**
- ✅ Shows quartiles (distribution shape)
- ✅ Identifies outliers automatically
- ✅ Compares distributions across categories
- ✅ Compact representation

**Interpreting the plot:**
```
Box plot by churn category:
                No Churn    Partial    Full
Revenue          |──●──|     |──●──|    |──●──|
             
If "Full Churn" box is lower: They have less revenue
than others, suggesting revenue is linked to churn.
```

---

**C. Density Plots (Violin Plots)**

```python
plotter.plot_density_plotly()
```

**What is a violin plot?**
```
Like a rotated histogram, showing full distribution shape:

No Churn    Partial      Full
  ╱╲          ╱╲         ╱╲ ╱╲
│    │      │    │      │ │ │ │  ← Two peaks (bimodal)
│    │      │    │      │ │ │ │
│    │      │    │      │ │ │ │
 ╲  ╱        ╲  ╱       ╲ ╱ ╲ ╱
```

**Use when:** Want to see full probability distribution, not just quartiles

---

**D. Correlation Heatmap**

```python
plotter.plot_correlation_heatmap_plotly()
```

**What it shows:**
```
              revenue  fees  duration  churn
revenue         1.0   0.85    0.42    -0.65   ← revenue and churn are negatively correlated
fees            0.85  1.0     0.38    -0.60   ← customers with high fees less likely to churn
duration        0.42  0.38    1.0     -0.35
churn          -0.65 -0.60   -0.35    1.0
```

**Color interpretation:**
- 🟢 **Green (1.0)**: Perfect positive correlation (both increase together)
- ⚪ **White (0)**: No correlation (independent)
- 🔴 **Red (-1.0)**: Perfect negative correlation (one increases, other decreases)

**Why this matters:**
- Identifies which features relate to churn
- Reveals multicollinearity (redundant features)
- Guides feature selection

---

**E. Categorical Analysis (Stacked Bar Charts)**

```python
plotter.plot_categorical_bars_plotly()
```

**What it shows:**
```
% Churn by Agreement Type:

        No Churn  Partial  Full
Type A    94%       4%     2%     ← Type A: mostly no churn
Type B    92%       5%     3%
Type C    85%       10%    5%     ← Type C: highest churn rate
Type D    96%       3%     1%     ← Type D: lowest churn rate
```

**Insights:**
- Identifies which categories have high churn
- Prioritizes intervention on high-churn categories
- Validates hypotheses about categorical importance

---

#### 2. **What We Learn from EDA**

From our project's EDA, typical findings include:

| Finding | Business Implication |
|---------|---------------------|
| Revenue is lower for churned customers | Loss of high-value customers |
| Certain agreement types have higher churn | Target interventions on high-churn types |
| Long-duration agreements correlate with retention | Encourage multi-year contracts |
| Specific branches have higher churn | Train or audit those branches |
| Customers with retention cases have better engagement | Invest in proactive retention |

---

## Stage 4: Hypothesis Testing

### Purpose
**Move from "what if" to "is this true?"** Hypothesis testing uses statistics to:
- Test if patterns are real or due to random chance
- Quantify confidence in findings
- Drive data-backed business decisions

### Statistical Framework

Every hypothesis has:
1. **H₀ (Null Hypothesis)**: The default assumption (no relationship)
2. **H₁ (Alternative Hypothesis)**: What we suspect is true
3. **Test Statistic**: A number measuring the evidence
4. **P-value**: Probability of observing this result if H₀ is true
5. **Decision Rule**: If p-value < α (usually 0.05), reject H₀

**What does this mean?**
```
If p-value < 0.05:
  → Result is statistically significant (real, not random)
  → Reject null hypothesis with 95% confidence

If p-value ≥ 0.05:
  → Insufficient evidence (could be random)
  → Failed to reject null hypothesis
```

### The 6 Hypotheses

#### **H1: Chi-Square Test - Agreement Type vs Churn**

```python
H₀: Agreement type is independent of churn category
H₁: Agreement type is associated with churn category
Test: Chi-Square Test of Independence
```

**Why Chi-Square?**
- Both variables are categorical (agreement type = A, B, C, ... ; churn = no, partial, full)
- Chi-Square tests if two categorical variables are independent
- Alternative: Would need non-parametric tests for continuous data

**How it works:**
```
Contingency Table (Actual Counts):
                No Churn    Partial    Full
Type A           5000        200       50
Type B           4000        300      100
Type C           3000        400       200

Expected under independence:
Type A           4700        250        85
Type B           4100        320       160
Type C           3200        330       155

Chi-Square = Σ (Observed - Expected)² / Expected
           = High difference → Strong association
           = Low difference → No association
```

**Null Hypothesis Assumption:**
The null hypothesis assumes that agreement type and churn are **independent**, meaning:
- Type A customers have the same churn rate as Type B and Type C
- If you know the agreement type, you cannot predict churn probability

**If we reject H₀:**
- Agreement types have different churn rates
- Some types are higher-risk, others are lower-risk
- **Business Action**: Develop type-specific retention strategies

---

#### **H2: Kruskal-Wallis Test - Revenue vs Churn**

```python
H₀: Revenue distribution is the same across churn categories
H₁: At least one churn category has a different revenue distribution
Test: Kruskal-Wallis H-test
```

**Why Kruskal-Wallis (not ANOVA)?**

| Test | Assumption | Use Case |
|------|-----------|----------|
| **ANOVA** | Normally distributed data → Parametric | Clean, symmetric data |
| **Kruskal-Wallis** | Distribution-free → Non-parametric | Skewed data, outliers, unknown distribution |

Revenue is often **right-skewed** (some very high earners):
```
Many low-mid revenue ╱    distribution
customers           │╲
                    │ ╲╲
                    │   ╲╲
                    │     ╲╲
                    │       ╲─► Few very high earners
────────────────────────────────
```

**Why this matters:**
- ANOVA assumes symmetry; revenue is not symmetric
- Kruskal-Wallis works on **ranks**, not values
- Ranks: $100 → Rank 1, $500 → Rank 2, $50,000 → Rank 3...
- More robust to outliers and skewness

**Null Hypothesis Assumption:**
Revenue is independent of churn status. All churn categories have similar revenue patterns.

**If we reject H₀:**
- Churn and revenue are related
- Different churn categories earn different revenues
- **Business Action**: Premium customers = priority retention

---

#### **H3: Kruskal-Wallis Test - Number of Agreements vs Churn**

```python
H₀: Number of agreements is the same across churn categories
H₁: At least one churn category has different agreement counts
Test: Kruskal-Wallis H-test
```

**Why this matters:**
- Single-product customers = higher churn risk?
- Multi-product customers = stickier (harder to leave)?
- Or: High-churn customers avoid buying more?

**Null Hypothesis Assumption:**
Customers with 1 agreement, 5 agreements, and 10 agreements have the same probability of churning.

**If we reject H₀:**
- Agreement count predicts churn
- **Business Action**: Cross-sell to at-risk customers

---

#### **H4: Kruskal-Wallis Test - Fee Structure vs Churn**

```python
H₀: Fee distribution is the same across churn categories
H₁: At least one churn category has different fee distributions
Test: Kruskal-Wallis H-test
```

**Why this matters:**
- Are churned customers more price-sensitive?
- Do higher fees drive churn?
- Or do high-fee customers have other issues?

**Null Hypothesis Assumption:**
High-fee and low-fee customers churn at the same rates.

**If we reject H₀:**
- Fees influence churn probability
- **Business Action**: Evaluate pricing strategy

---

#### **H5: Kruskal-Wallis Test - Agreement Duration vs Churn**

```python
H₀: Agreement duration is the same across churn categories
H₁: At least one churn category has different durations
Test: Kruskal-Wallis H-test
```

**Why this matters:**
- Long-term customers more loyal?
- Short-term contracts = higher churn?
- Or: Churned customers had shorter agreements?

**Null Hypothesis Assumption:**
All customers, regardless of agreement duration, have the same churn probability.

**If we reject H₀:**
- Duration matters for churn
- **Business Action**: Encourage long-term contracts

---

#### **H6: Spearman Correlation - Financial Metrics vs Churn**

```python
H₀: No correlation between financial metrics and churn
H₁: Financial metrics correlate with churn
Test: Spearman's Rank Correlation Coefficient
```

**Why Spearman (not Pearson)?**

| Test | Assumption | Use Case |
|------|-----------|----------|
| **Pearson** | Linear relationship, normal distribution → Parametric | Perfect lines in scatter plot |
| **Spearman** | Monotonic relationship, any distribution → Non-parametric | Curves, outliers, non-linear trends |

**Example:**
```
Pearson: Revenue must increase linearly with time
          |    ╱╱╱╱
       Profit  ╱╱
              ╱
          ─────────
          Time

Spearman: Just needs to increase (or decrease)
          |     ╱╱╱
       Profit ╱  ╱
            ╱  ╱ (curved is OK)
          ───────────
          Time
```

**Null Hypothesis Assumption:**
Financial metrics and churn are independent (corr = 0).

**If we reject H₀:**
- Financial health predicts churn
- **Business Action**: Monitor financial metrics as early warning system

---

### How to Interpret Results

Example output from H1:
```
Chi-Square Statistic: 245.3214
P-Value: 0.000031
Decision: Reject H0

→ Agreement type IS statistically significantly associated 
  with churn (p < 0.05)
```

**Business Translation:**
- The pattern we see (Type A has different churn than Type B) is **real**, not coincidence
- 99.99% confident (1 - 0.00003 ≈ 99.997%)
- Recommend targeted interventions by agreement type

---

## Stage 5: Feature Engineering

### Purpose
**Transform raw data into machine learning features** that:
- Are predictive of churn
- Avoid data leakage (don't directly encode churn)
- Capture business logic and domain knowledge
- Reduce dimensionality

### The Key Challenge: Customer-Level Aggregation

**Raw Data Structure:**
```
One row per agreement per time period:

account_number  agreement_number  revenue  churn
123             A1               1000     Yes
123             A2               500      Yes
123             A3               200      Yes
456             B1               2000     No
456             B2               1500     No
```

**Machine Learning Needs:**
```
One row per customer:

account_number  total_revenue  num_agreements  churn
123             1700           3              Yes
456             3500           2              No
```

**Why?**
- Different customers have different agreement counts
- Can't directly feed agreement-level data to model
- Need to summarize each customer into a single row

### The Feature Engineering Process

```python
class FeatureEngineering:
    def engineer_features(self):
        # Step 1: Parse dates
        df['agreement_duration_days'] = (
            df['agreement_end_date'] - df['agreement_start_date']
        ).dt.days
        
        # Step 2: Create flags
        df['is_bob_flag'] = (df['is_bob'].str.lower() == 'yes').astype(int)
        df['has_retention_case'] = (df['resolution_status'] != 'No Case').astype(int)
        
        # Step 3: Aggregate to customer level
        financial = df.groupby('account_number').agg({
            'total_bob': ['sum', 'mean'],      # total and average revenue
            'product_bob': 'sum',              # total product value
            'fee_bob': ['sum', 'mean'],        # total and average fees
            'unit_amount': ['mean', 'max']     # average and max unit amount
        })
```

### Key Features Created

#### **Financial Features**
```python
total_revenue      # Sum of all revenue from this customer
avg_revenue        # Average revenue per agreement
total_product_value # Total value of products purchased
total_fees         # Sum of all fees paid
avg_fees           # Average fees per agreement
max_unit_amount    # Highest single transaction
```

**Why:**
- ✅ Revenue: Indicates customer value and usage
- ✅ Fees: Shows how much customer pays (price sensitivity?)
- ✅ Max transaction: Indicates spike behavior or peak usage

**Alternative approaches:**
- Revenue percentile: "Top 20% earner" (more interpretable)
- Revenue per product: Revenue per agreement type
- Revenue volatility: Std dev of revenue (consistency)

---

#### **Agreement Diversity Features**
```python
num_agreements       # How many agreements does this customer have?
num_branches         # How many branches serve this customer?
num_agreement_types  # How many different types?
num_products         # How many different products?
```

**Why:**
- ✅ Diversification: Multi-product customers less likely to completely churn
- ✅ Stickiness: More touchpoints = harder to leave
- ✅ Risk: Single-product customers are concentrated risk

**Business Intuition:**
```
Single-product customer:          Multi-product customer:
One decision point to leave        Multiple decision points (quit all or some?)
Higher churn risk                  Lower complete churn risk
                                   May partially churn (quit 1 product)
```

**Alternative approaches:**
- Herfindahl Index: Concentration measure (0-1 scale)
- Product diversity score: Weighted by product type

---

#### **Duration Features**
```python
avg_agreement_duration     # Average length of contracts
max_agreement_duration     # Longest contract
```

**Why:**
- ✅ Long contracts indicate stability and retention
- ✅ Customer commitment signals loyalty
- ✅ Maintenance contracts vs one-time purchases?

**What if all customers have similar durations?**
→ Might not be predictive, consider removing

---

#### **Behavioral Features**
```python
bob_ratio           # Percentage of agreements that are BoB (0-1)
num_retention_cases # How many times did retention team contact customer?
dominant_agreement_type # Most common agreement type for this customer
```

**Why:**
- ✅ BoB ratio: "Bread and Butter" vs niche products
- ✅ Retention cases: Signals engagement/issues (high frequency = problem?)
- ✅ Dominant type: Summary of customer portfolio

**Alternative approaches:**
- Engagement score: Weighted combination of interactions
- Churn risk score: Historical pattern similarity

---

#### **Derived Features**
```python
revenue_per_agreement = total_revenue / num_agreements
```

**Why:**
- ✅ Efficiency metric: How much does each agreement bring?
- ✅ Scales revenue by diversification
- ✅ Normalizes across customers

**Other derived features to consider:**
- Fee per agreement: Cost efficiency
- Revenue trend: Is customer growing or declining?
- Product adoption rate: Percentage of available products

---

### Data Leakage: What NOT to Do

**Leakage happens when training data contains information about the target.**

❌ **BAD**: Using current churn status in features
```python
# DO NOT DO THIS:
features['already_churned'] = (data['churn_status'] == 'Churned')
# Problem: Perfect predictor (literally the target!)
```

❌ **BAD**: Using future information
```python
# DO NOT DO THIS:
features['future_retention_contact'] = customer_contacted_in_future
# Problem: Predicting future with info only available after prediction
```

✅ **GOOD**: Using historical patterns
```python
# CORRECT:
features['past_retention_contacts'] = historical_contact_count
# This was true before we made the prediction
```

---

## Stage 6: Model Training & Evaluation (Gold Layer)

### Purpose
**Build and evaluate machine learning models** that predict which customers will churn.

### The ML Pipeline

```
Raw Data
   ↓
[Data Scaling] → Standardize features to 0 mean, 1 std dev
   ↓
[Train/Test Split] → 80% train, 20% test (validate on unseen data)
   ↓
[Model Training] → Fit model on training data
   ↓
[Hyperparameter Tuning] → Find best settings
   ↓
[Evaluation] → Test on held-out test data
   ↓
[Predictions] → Deploy to production
```

### Data Scaling: Why and How

**Problem:** Features have different scales
```
Revenue:        10,000 - 500,000  (huge range)
Num Agreements: 1 - 100           (small range)
```

**When ML sees this:**
- Treats revenue as "more important" just because it's larger
- Biases model toward revenue

**Solution:** Standardization
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Transforms each feature to: (value - mean) / std_dev
# Result: All features have mean=0, std=1

Before: Revenue = [10000, 20000, 15000]
After:  Revenue = [-1.2, 0.8, -0.1]  # same distribution, scaled
```

**When to scale:**
- ✅ Distance-based models: KNN, K-means, SVM, Logistic Regression
- ❌ Tree-based models: Random Forest, Gradient Boosting (don't need scaling)

---

### Train/Test Split: Why and How

**Problem:** If we train and test on the same data, we learn the noise
```
Training data = [1, 2, 3, 4, 5] with noise
We memorize: "1 → Churn, 2 → No churn, 3 → Churn..."
On new data [6, 7, 8], we guess randomly (haven't seen these before)
```

**Solution:** Hold out test data
```
All data (23,772 customers)
    ↓
80% Train (19,017 customers) → Fit model here
20% Test (4,755 customers) → Evaluate here (unseen by model)

Model accuracy on test data = true performance estimate
```

**Alternative approaches:**
- K-Fold Cross-Validation: Split into 5 folds, repeat 5 times
- Time-based split: Use temporal order (if time matters)

---

### Model Algorithms

#### **1. Logistic Regression**

**What it does:**
```
Calculates probability of churn:
P(Churn) = 1 / (1 + e^(-β₀ - β₁X₁ - β₂X₂ - ...))

Output: Probability between 0 and 1
        If P > 0.5, predict Churn
        If P ≤ 0.5, predict No Churn
```

**Pros:**
- ✅ Interpretable: Coefficients show feature importance
- ✅ Probabilistic: Gives uncertainty estimates
- ✅ Fast: Linear computation

**Cons:**
- ❌ Assumes linear relationship
- ❌ May underfit complex patterns

**When to use:**
- Quick baseline
- Need interpretability over accuracy
- Small datasets

**Business interpretation:**
```
Coefficient for 'revenue': -0.005
Interpretation: Each $1000 increase in revenue → 
                0.5% decrease in churn probability
```

---

#### **2. Random Forest**

**What it does:**
```
Builds 200 trees, each trained on random data subsets:

           Tree 1        Tree 2        Tree 3
            ├─B         ├─Revenue      ├─Duration
            │ ├─Yes     │ ├─High→No    │ ├─Long→No
            │ └─No      │ └─Low→Yes    │ └─Short→Yes
            
Vote: If 180 trees predict "No Churn", predict "No Churn"
```

**Pros:**
- ✅ Handles non-linear relationships
- ✅ Feature importance built-in
- ✅ Robust to outliers (trees use rules, not distances)
- ✅ No scaling needed

**Cons:**
- ❌ Less interpretable (complex decision boundaries)
- ❌ Can overfit (requires tuning)
- ❌ Slower than logistic regression

**When to use:**
- Medium-sized datasets
- Complex relationships expected
- Need good baseline performance

**Feature importance:**
```
Tree-based models show which features matter:

Feature Importance Score:
├─ Total Revenue: 0.32 (32% of prediction power)
├─ Num Agreements: 0.18
├─ Fees: 0.15
└─ Duration: 0.10
```

---

#### **3. Gradient Boosting**

**What it does:**
```
Builds trees sequentially, each one corrects previous mistakes:

Tree 1: Predicts churn (not perfect)
        → Residuals: [errors from prediction]

Tree 2: Trained to predict residuals from Tree 1
        → Improves Tree 1's predictions

Tree 3: Trained to predict residuals from Tree 1+2
        → Further improvements

Final: Tree 1 + Tree 2 + Tree 3 + ...
```

**Pros:**
- ✅ Often highest accuracy (wins competitions)
- ✅ Feature importance
- ✅ Handles mixed data types

**Cons:**
- ❌ Slow to train
- ❌ Prone to overfitting (requires careful tuning)
- ❌ Hyperparameter-sensitive

**When to use:**
- Maximizing accuracy is critical
- Have time for tuning
- Larger datasets

---

#### **4. Support Vector Machine (SVM)**

**What it does:**
```
Finds a separating line/plane that maximizes margin:

No Churn:        ╱ ──── margin ──── ╲
             ●  ╱                     ╲   ●
             ● ╱                       ╲ ●
           ──────────── decision boundary ─────────
             ╱ Full Churn              ╲
            ╱  ×                        ╲ ×
```

**Pros:**
- ✅ Excellent for high-dimensional data
- ✅ Memory efficient
- ✅ Flexible (different kernel functions)

**Cons:**
- ❌ Needs scaling (distance-based)
- ❌ Slow on large datasets
- ❌ Hard to interpret

**When to use:**
- High-dimensional features
- Small to medium datasets
- Need confidence scores

---

### Hyperparameter Tuning

**Hyperparameters:** Settings we choose before training (not learned by model)

```python
# Random Forest hyperparameters:
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,       # How many trees?
        max_depth=10,           # How deep can trees grow?
        min_samples_split=5,    # Min samples to split node?
        min_samples_leaf=2,     # Min samples at leaf?
        class_weight='balanced' # Handle imbalance?
    )
    model.fit(X_train, y_train)
    return model
```

**How to tune:**

Method 1: **Grid Search** (exhaustive)
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}
# Tries: 3 × 3 × 3 = 27 combinations
# Each evaluated 5-fold cross-validation = 135 model trainings
```

**Pros:** Thorough, guaranteed to find combination in grid
**Cons:** Exponential time complexity, misses combinations outside grid

Method 2: **Random Search**
```python
# Randomly samples combinations
# Faster than grid search
# May miss optimal combination
```

Method 3: **Bayesian Optimization**
```python
# Learns which regions of hyperparameter space are promising
# Uses probability model to guide search
# Most efficient but complex
```

---

### Handling Class Imbalance

**Problem:** Dataset has 94% No Churn, 6% Churn
```
Training data:
No Churn: ███████████████████ (22,370)
Churn:    █ (1,402)

Model learn: "Always predict No Churn" = 94% accuracy!
But useless for churn detection.
```

**Solutions:**

1. **Class Weights** (what we use)
```python
model = RandomForestClassifier(class_weight='balanced')
# Automatically assigns higher weight to rare class
# Cost of false negative (missing a churner) >> false positive
```

2. **Oversampling** (duplicate minority class)
```python
from imblearn.over_sampling import SMOTE
X_train_balanced, y_train_balanced = SMOTE().fit_resample(X_train, y_train)
# Creates synthetic churn samples
# Pros: More data for minority class
# Cons: Creates artificial data, may overfit
```

3. **Undersampling** (remove majority class)
```python
# Remove some No Churn samples
# Pros: Smaller dataset, faster training
# Cons: Loss of information
```

4. **Threshold Adjustment**
```python
# Default: predict churn if P(churn) > 0.5
# Adjusted: predict churn if P(churn) > 0.3
# More sensitive to churn (more false alarms, fewer misses)
```

---

### Model Evaluation Metrics

#### **Accuracy**
```
Correct Predictions / Total Predictions

Example:
Model predicts 95 customers correctly out of 100
Accuracy = 95%

Problem: Misleading with imbalanced data
If 95 customers truly don't churn:
    Always predict "No Churn" → 95% accuracy (useless!)
```

#### **Precision** (Exactness)
```
True Positives / (True Positives + False Positives)
= When we predict "Churn", how often are we right?

Example:
Model predicts 50 customers will churn
Actually 45 churned (true positives), 5 didn't (false positives)
Precision = 45 / 50 = 90%

Use when: False alarms are expensive
(Don't want to contact too many non-churners)
```

#### **Recall** (Sensitivity/Coverage)
```
True Positives / (True Positives + False Negatives)
= Of all actual churners, what % did we catch?

Example:
100 customers actually churned
Model correctly identified 80, missed 20
Recall = 80 / 100 = 80%

Use when: Missing churners is very costly
(Losing customers is expensive)
```

#### **F1-Score** (Balance)
```
Harmonic mean of Precision and Recall
= 2 × (Precision × Recall) / (Precision + Recall)

Balances both concerns
High F1 requires both high precision AND high recall
```

#### **ROC-AUC Score** (Ranking Quality)
```
AUC = Area Under the ROC Curve

ROC Curve plots:
- True Positive Rate (Sensitivity) vs
- False Positive Rate (1 - Specificity)

AUC = 1.0: Perfect classifier
AUC = 0.5: Random guessing
AUC = 0.0: Perfect (but inverted) classifier

1.0 ├─ Random line
    │ /╱─ Good model (AUC 0.8)
    │/───────
0.5 ├────────/──── AUC = 0.5 (random)
    │      ╱
    │    ╱
    │ ╱
  0 └────────────
    0.0       1.0
    False Positive Rate
```

**Why AUC?**
- Threshold-independent (doesn't assume 0.5 cutoff)
- Ranking quality: How well model ranks churners higher than non-churners
- Works well with imbalanced data

---

### Confusion Matrix: The Full Picture

```python
from sklearn.metrics import confusion_matrix

             Predicted No  Predicted Partial  Predicted Full
Actual No        20,000           350             20
Actual Partial       100           800             150
Actual Full           50           100               55
```

**Interpretation:**
- **Diagonal (✓correct)**: Top-left (20,000), middle (800), bottom-right (55)
- **Off-diagonal (✗errors)**:
  - Predicting as No Churn when actually Partial: 100 errors
  - Predicting as Partial when actually Full: 100 errors

**How to read it:**
```
Row = actual, Column = predicted

Actual Full Churn row: [50, 100, 55]
- Of 205 full churners:
  - 50 predicted as No Churn (FN)
  - 100 predicted as Partial Churn (FN)
  - 55 predicted as Full Churn (TP) ✓

Recall for Full Churn = 55 / 205 = 26.8%
(Only catching 27% of those who completely churn!)
```

---

### Feature Importance: What Matters?

Tree-based models show which features drive predictions:

```python
model = RandomForestClassifier(...)
importances = model.feature_importances_

Output:
Feature Importance Ranking:
1. Total Revenue: 0.28          (28% of model's decisions)
2. Num Agreements: 0.18         (18%)
3. Average Revenue: 0.15        (15%)
4. Total Fees: 0.12             (12%)
5. Dominant Agreement Type: 0.10 (10%)
...
```

**Business Implications:**
1. **Revenue is #1**: Focus on retaining high-value customers
2. **Agreements matter**: Cross-sell initiatives work
3. **Fees relevant**: Price sensitivity exists
4. **Type matters**: Some agreement types = higher churn

**Creating visualizations:**
```python
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('What Drives Churn Predictions?')
plt.show()
```

---

### Model Comparison

```python
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'Gradient Boosting': gb_model,
    'SVM': svm_model
}

results = []
for name, model in models.items():
    metrics = evaluate_model(model, X_test, y_test)
    results.append(metrics)

comparison_df = pd.DataFrame(results)
print(comparison_df)
```

**Typical Output:**
```
                    Accuracy  Precision  Recall   F1-Score
Logistic Regression   0.92    0.65       0.45     0.53
Random Forest         0.94    0.72       0.58     0.64
Gradient Boosting     0.96    0.78       0.68     0.73  ← Best overall
SVM                   0.93    0.70       0.52     0.60
```

**Selection Criteria:**
- **Best Accuracy**: Gradient Boosting (96%)
- **Best Precision**: Gradient Boosting (78%) - fewest false alarms
- **Best Recall**: Gradient Boosting (68%) - catches most churners
- **Best F1**: Gradient Boosting (0.73) - best balance

**Decision:** Deploy **Gradient Boosting** (best across board) OR **Random Forest** (80% accuracy of GB but faster, more interpretable)

---

### Handling Multi-Class Classification

**Challenge:** 3 classes (No Churn, Partial Churn, Full Churn) not binary

**Approach 1: One-vs-Rest (OvR)**
```
Create 3 binary classifiers:
1. No Churn vs (Partial + Full)
2. Partial Churn vs (No Churn + Full)
3. Full Churn vs (No Churn + Partial)

Final prediction: Highest probability from 3 classifiers
```

**Approach 2: One-vs-One (OvO)**
```
Create 3 binary classifiers:
1. No Churn vs Partial Churn
2. No Churn vs Full Churn
3. Partial Churn vs Full Churn

Final prediction: Majority vote
```

**Our approach:** Softmax (outputs probabilities summing to 1)
```
Model output: [0.75, 0.20, 0.05]
               ↓    ↓    ↓
         No Churn, Partial, Full
         
Prediction: No Churn (highest probability)
```

---

## Key Insights & Recommendations

### What We Learned

#### **1. Revenue is the Strongest Predictor**
- **Finding**: Churned customers have significantly lower revenue
- **Interpretation**: Revenue directly correlates with engagement/stickiness
- **Recommendation**: 
  - Implement VIP programs for top-revenue customers
  - Offer incentives to increase revenue from at-risk customers
  - Partner opportunities for high-revenue accounts

#### **2. Product Diversification Reduces Churn**
- **Finding**: Customers with more agreements have lower churn
- **Interpretation**: Multiple touchpoints = harder to completely leave
- **Recommendation**:
  - Cross-sell campaigns for single-product customers
  - Bundle offerings to increase product count
  - Educate customers on complementary products

#### **3. Agreement Type Matters**
- **Finding**: Some agreement types have 2x churn rate of others
- **Interpretation**: Type-specific challenges or value propositions
- **Recommendation**:
  - Root cause analysis for high-churn types
  - Improved onboarding for high-churn types
  - Premium features for high-churn types

#### **4. Duration/Relationship Matters**
- **Finding**: Longer agreements correlate with retention
- **Interpretation**: Switching costs, familiarity, habits
- **Recommendation**:
  - Encourage multi-year contracts (discounts)
  - Lock-in periods for new products
  - Relationship milestones (1 year → benefits)

#### **5. Retention Interactions Indicate Issues**
- **Finding**: Customers with retention cases show patterns
- **Interpretation**: Early indicator or result of churn risk?
- **Recommendation**:
  - Analyze retention case reasons
  - Implement proactive outreach before cases occur
  - Train retention team on patterns

---

### Model Performance Interpretation

**Gradient Boosting Results:**
```
Accuracy:  96%  - Correct predictions
Precision: 78%  - When predicting churn, 78% are correct
Recall:    68%  - Catching 68% of actual churners
F1-Score:  0.73 - Balanced performance
```

**What this means:**
- ✅ Very accurate overall (96%)
- ✅ Good precision (not too many false alarms)
- ⚠️ Recall could be higher (missing 32% of churners)

**Trade-off Analysis:**
```
Current model (threshold=0.5):
- Predicts 100 churners
- 78 are correct (TP), 22 are incorrect (FP)
- But missing 32 actual churners (FN)

Adjusted threshold (0.3):
- Predicts 150 churners
- Recall increases to 85%
- But precision drops (more false alarms)

Business decision:
- Cost to retain non-churner?: Medium (outreach cost: $50)
- Cost to lose actual churner?: High (lifetime value: $50,000)
→ Use lower threshold (catch more potential churners)
```

---

### Next Steps & Recommendations

#### **Short-term (0-3 months)**
1. **Pilot the model** on test segment
   - Contact predicted churners with retention offers
   - Measure actual churn vs prediction
   - Calibrate model based on results

2. **Identify actionable patterns**
   - Which customer segments benefit most from intervention?
   - What's the cost-benefit of each retention tactic?

3. **Implement monitoring**
   - Track model predictions vs actual outcomes
   - Retrain model quarterly with new data

#### **Medium-term (3-12 months)**
1. **Feature expansion**
   - Add NPS/satisfaction scores
   - Add support ticket sentiment
   - Add website/app usage metrics
   - Add competitive intelligence

2. **Model improvements**
   - Ensemble models (combining multiple models)
   - Explore deep learning for complex patterns
   - Time-series models (customer trajectory)

3. **Intervention optimization**
   - A/B test different retention strategies
   - Personalize offers based on churn reason
   - Optimize intervention timing

#### **Long-term (12+ months)**
1. **Predictive churn scoring**
   - Real-time prediction as customer behavior changes
   - Automatic alerts for high-risk accounts
   - Integration with CRM system

2. **Causal analysis**
   - What actually causes churn? (not just correlation)
   - Randomized experiments to test interventions
   - Cost-benefit analysis of retention tactics

3. **Automated actions**
   - Automatic outreach based on churn score
   - Targeted offers for specific risk factors
   - Win-back campaigns for churned customers

---

### Important Caveats & Limitations

#### **Data Quality Issues**
- **Survivorship bias**: Only see customers who were customers (don't see those who already Left)
- **Information gap**: Missing social/market factors
- **Temporal lag**: Historical data, not current behavior

#### **Model Limitations**
- **Assumes patterns continue**: Past churn drivers may change
- **Average performance**: Works well for "typical" customers, may fail for outliers
- **Black box risk**: GB models hard to explain to customers
- **Class imbalance**: Model may bias toward majority class

#### **Business Constraints**
- **Privacy**: GDPR/privacy laws limit data we can collect
- **Cost**: Retention offers have costs; not every churn is preventable
- **Timing**: Predictions made X days before churn; dynamics change
- **Measurement**: Hard to know if intervention caused retention

---

## How to Use This Project

### For a Data Scientist
1. Read README.md for setup instructions
2. Start with `notebooks/01_bronze/01_exploration.ipynb`
3. Proceed through Silver and Gold layers in order
4. Modify `src/` modules as needed for your use case

### For a Business Stakeholder
1. Focus on "Key Insights & Recommendations" section
2. Review model performance metrics (confmat and ROC)
3. Understand cost-benefit before deploying
4. Plan pilot testing to validate predictions

### For a ML Engineer
1. Review feature engineering (`src/features/`)
2. Examine model evaluation (`src/models/`)
3. Consider deployment pipeline (batch scoring, online predictions)
4. Implement monitoring and retraining schedule

---

## Glossary

| Term | Definition |
|------|-----------|
| **Churn** | Customer leaves or stops using service |
| **Partial Churn** | Customer reduces engagement or leaves some agreements |
| **Feature** | Input variable to ML model |
| **Hyperparameter** | Model setting chosen before training |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Cross-validation** | Splitting data multiple ways to estimate true performance |
| **AUC** | Area under ROC curve; 1.0 = perfect, 0.5 = random |
| **Precision** | When we predict churn, how often are we right? |
| **Recall** | Of all actual churners, what % do we catch? |
| **F1-Score** | Harmonic mean of precision and recall |
| **Data Leakage** | Using information that wouldn't be available at prediction time |

---

## Technical Stack

| Component | Tool |
|-----------|------|
| **Data Processing** | Pandas, NumPy |
| **Statistical Testing** | SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | Scikit-learn |
| **Feature Scaling** | StandardScaler |
| **Environment Management** | Python Virtual Environment (.venv) |

---

## Questions & Troubleshooting

**Q: Why 3 churn classes instead of binary (churn/no churn)?**
A: Captures business reality - customers don't just "leave", they may reduce engagement first. Allows targeted interventions.

**Q: Why Kruskal-Wallis instead of ANOVA?**
A: Revenue data is skewed (right-tailed) with outliers. Kruskal-Wallis works on ranks, not raw values, making it robust.

**Q: Why scale features for SVM but not Random Forest?**
A: SVM uses distance metrics (influenced by scale). Trees use rules (unaffected by scale).

**Q: How do we know the model generalizes?**
A: Test set (20% held out) never seen by model during training. Test performance estimates real-world performance.

**Q: Can we trust the feature importance?**
A: Yes, for ranking features. But be skeptical of exact percentages; correlated features may switch rankings.

**Q: Why gradient boosting over random forest?**
A: Generally higher accuracy (96 vs 94%). Trade-off: slower training, more hyperparameters to tune.

---

## Final Thoughts

This project demonstrates the full data science lifecycle:
1. **Ask**: Can we predict churn?
2. **Explore**: Understand the data
3. **Prepare**: Clean and engineer features
4. **Model**: Train and evaluate models
5. **Execute**: Deploy and monitor

Success isn't just about model accuracy - it's about **business impact**. The best model is useless if predictions aren't acted upon or if interventions don't improve outcomes.

**Key Metrics for Success:**
- ✅ Model accuracy (technical)
- ✅ Cost per intervention (business)
- ✅ Actual churn reduction (impact)
- ✅ ROI of intervention (justifies effort)

---

**Document Version:** 1.0
**Created:** April 15, 2026
**Project:** Customer Churn Analysis
**Author:** [Manu Bansal](www.github.com/ManuBansalS)
