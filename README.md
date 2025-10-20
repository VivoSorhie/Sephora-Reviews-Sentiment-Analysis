# Sephora Product Risk Mitigation via NLP Sentiment Analysis

A deep dive into 100,000+ customer reviews to quantify financial risk and drive business strategy using a state-of-the-art DistilBERT model.

[![View Project Notebook](https://img.shields.io/badge/View-Project_Notebook-blue?style=for-the-badge&logo=jupyter)]([your-github-username]/[your-repo-name]/blob/main/sephora_sentiment_analysis.ipynb)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Key Findings & Actionable Insights](#key-findings--actionable-insights)
  - [1. Strategic Risk Matrix](#1-strategic-risk-matrix-diagnosing-product-failures)
  - [2. Financial Risk by Customer Concerns](#2-financial-risk-by-customer-concerns)
  - [3. The Financial Engine](#3-the-financial-engine-mid-range-is-the-market-leader)
  - [4. The Myth of Price vs. Quality](#4-the-myth-of-price-vs-quality-is-busted)
  - [5. Ingredient Connection](#5-ingredient-connection-data-driven-formulation)
- [How to Run This Project](#how-to-run-this-project)


---

## Project Overview
This project transforms unstructured Sephora customer reviews into a powerful strategic asset. The primary objective was to move beyond simple star ratings and use Natural Language Processing (NLP) to understand the *why* behind customer satisfaction and dissatisfaction. By building a high-performance sentiment classification model, this analysis quantifies product risk, uncovers key drivers of customer sentiment, and provides a series of data-driven recommendations for Marketing, R&D, and Product teams.

---

## Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFD21E?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-88d9de?style=for-the-badge&logo=seaborn&logoColor=white)

---

## Methodology
1. **Data Ingestion & Cleaning:** Loaded and merged multiple datasets, then performed rigorous text preprocessing (lemmatization, stopword removal) to create a clean corpus for analysis.  
2. **Modeling:**
   - Established a **Logistic Regression** baseline with TF-IDF features.
   - Fine-tuned a state-of-the-art **DistilBERT** transformer model for high-precision sentiment classification.  
3. **Deep-Dive Analysis:** Used the validated model and cleaned data to conduct a series of strategic deep dives into the drivers of business value and risk.

---

## 🤖 Model Performance
- **Accuracy:** **98%**  
- **AUC Score:** **0.99**  
- **Weighted F1-Score:** **0.98**  
- **Precision (negative reviews):** 0.92  
- **Recall (negative reviews):** 0.92  

![ROC-AUC Curve](roc_auc_curve.png.png)

---

## Key Findings & Actionable Insights

### 1. Strategic Risk Matrix: Diagnosing Product Failures
- Products plotted by **failure rate vs. financial impact**.
- **Critical Priority:** High-revenue, high-failure products (e.g., *Protini Polypeptide Moisturizer*).  
- **Protect the Stars:** High-revenue, low-failure products.  

**Actionable Insight:** R&D and Quality Control must investigate **Critical Priority** products immediately.  

![Strategic Risk Matrix](strategic_risk_matrix.png.png)

---

### 2. Financial Risk by Customer Concerns
- **Highest Revenue Risk:** Acne-related complaints → **\$140,000+** estimated revenue risk.  
- **Other Notable Risks:** Sensitivity and pore-related complaints → moderate but significant revenue risk.  

**Actionable Insight:**  
- **R&D:** Reformulate high-revenue products to reduce acne triggers, sensitivity, and pore issues.  
- **Marketing & Customer Support:** Proactively address concerns via education, ingredient transparency, and targeted campaigns.  

![Revenue Risk by Customer Concern](concerns_financial_risk.png)

---

### 3. The Financial Engine: Mid-Range is the Market Leader
- **Mid-Range price tier ($25-$75)** drives over **55% of total estimated revenue**.
- Strong combination of pricing and review volume makes this segment Sephora's core financial driver.  

**Actionable Insight:** Prioritize Mid-Range tier in marketing campaigns, inventory management, and customer acquisition strategies.  

![Revenue Contribution by Price Tier](revenue_treemap.png.png)

---

### 4. The Myth of Price vs. Quality is Busted
- **Price has near-zero correlation with customer satisfaction** across all tiers (Budget → Ultra-Luxury).  

**Actionable Insight:** Focus marketing on **tangible benefits, proven ingredients, and problem-solving capabilities**, not premium price as a proxy for quality.  

![Price vs Quality Violin Plot](price_quality_violin.png.png)

---

### 5. Ingredient Connection: Data-Driven Formulation
- **"Hero" Ingredients:** Hyaluronic Acid, Ceramides, Squalane → higher ratings.  
- **"Villain" Ingredient:** Alcohol → largest driver of dissatisfaction.  

**Actionable Insight:**  
- **R&D:** Prioritize hero ingredients, reduce/eliminate alcohol.  
- **Marketing:** Highlight data-backed attributes in campaigns.  

![Ingredient Impact on Ratings](ingredient_impact.png.png)

---

## 🚀 How to Run This Project
1. Clone the repository:
```bash
git clone [https://github.com/](https://github.com/)[your-github-username]/[your-repo-name].git
