# Promotion Effectiveness at a Fashion Retail Chain

---

#  B1. Problem Formulation (8 Marks)

## (a) Machine Learning Problem Formulation

The goal is to determine which promotion maximizes the number of items sold at each store.

- **Target Variable:**  
  `items_sold` (number of items sold per store per month)

- **Input Features:**  
  - Store characteristics: `store_size`, `location_type`, `store_id`  
  - Promotion: `promotion_type`  
  - Temporal features: `month`, `year`, `is_weekend`, `is_festival`  
  - Market conditions: `competition_density`, footfall, demographics  

- **Type of ML Problem:**  
  **Supervised Regression Problem**

- **Justification:**  
  The target variable is continuous (number of items sold), so regression is appropriate.  
  The model predicts expected sales under different conditions and promotions.

---

## (b) Why Use Items Sold Instead of Revenue

Using **items_sold** is more reliable than revenue because:

1. **Removes pricing bias:**  
   Revenue is affected by discounts and pricing strategies.

2. **Captures true demand:**  
   Items sold reflects customer response directly.

3. **Fair comparison across promotions:**  
   Promotions like BOGO may increase volume but reduce price per item.

### Broader Principle:
The target variable should **directly align with the business objective**.  
Choosing the wrong target can introduce bias and misleading results.

---

## (c) Alternative Modelling Strategy

Instead of one global model, use:

### **Segmented Models (Cluster-Based Approach)**

- Build separate models for:
  - Urban / Semi-urban / Rural stores
  - Small / Medium / Large stores

OR

### **Hierarchical Model (Multi-level Model)**

- Include store-level variations within a single model

### Justification:
- Customer behavior differs across locations
- Promotions perform differently across segments
- Improves model accuracy and personalization

---

#  B2. Data and EDA Strategy (10 Marks)

## (a) Data Joining and Dataset Design

### Tables:
- Transactions
- Store Attributes
- Promotion Details
- Calendar Table

### Joining Process:
- Join transactions with store attributes using `store_id`
- Join promotion details using `promotion_type`
- Join calendar using `transaction_date`

### Final Dataset Grain:
- **One row = one store per month**

### Aggregations:
- Total `items_sold` per store per month
- Monthly promotion applied
- Average competition density
- Festival/weekend indicators

---

## (b) Exploratory Data Analysis (EDA)

### 1. Promotion Performance Analysis
- **Chart:** Bar plot / boxplot  
- **Goal:** Compare average sales by promotion  
- **Impact:** Identify best-performing promotions  

---

### 2. Location-Based Sales Analysis
- **Chart:** Grouped bar chart  
- **Goal:** Compare urban vs rural sales  
- **Impact:** Justifies segmented models  

---

### 3. Time Trend Analysis
- **Chart:** Line plot (month vs sales)  
- **Goal:** Identify seasonality  
- **Impact:** Add time-based features  

---

### 4. Competition Impact
- **Chart:** Scatter plot  
- **Goal:** Relationship between competition and sales  
- **Impact:** Validate feature importance  

---

### 5. Festival & Weekend Impact
- **Chart:** Comparison plots  
- **Goal:** Identify demand spikes  
- **Impact:** Feature engineering  

---

## (c) Handling Promotion Imbalance

Since **80% data has no promotion**:

### Problem:
- Model becomes biased toward non-promotion cases
- Underestimates promotion effects

### Solutions:
- Use **class weighting or sampling**
- Oversample promotion data
- Evaluate performance separately for promotion vs non-promotion
- Consider modelling promotion impact separately

---

#  B3. Model Evaluation and Deployment (12 Marks)

## (a) Train-Test Split and Evaluation

### Train-Test Strategy:
- Use **time-based split**
- Train: First ~80% (earlier months)
- Test: Last ~20% (recent months)

### Why Not Random Split:
- Causes **data leakage**
- Future data influences training
- Unrealistic performance

---

### Evaluation Metrics:

1. **RMSE (Root Mean Squared Error):**
   - Penalizes large errors
   - Useful for business risk

2. **MAE (Mean Absolute Error):**
   - Average prediction error
   - Easy to interpret

3. **R² Score (optional):**
   - Measures explained variance

### Interpretation:
- Lower RMSE/MAE → better performance  
- Compare error with average sales volume  

---

## (b) Explaining Model Recommendations

To explain why different promotions are recommended:

### Step 1: Feature Importance
- Extract feature importance from model

### Step 2: Compare Conditions
- Compare December vs March:
  - Seasonal demand
  - Festival presence
  - Competition
  - Customer behavior

### Explanation to Business:
The model adapts to **seasonality and context**.  
Different promotions work better under different conditions.

---

## (c) Deployment Strategy

### 1. Save Model
```python
import joblib
joblib.dump(model, 'model.pkl')
