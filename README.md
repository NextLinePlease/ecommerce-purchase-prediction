# E-Commerce Purchase Prediction: End-to-End ML Pipeline & API

## Project Overview
An end-to-end machine learning system that predicts online shopping purchase intention from session-level behavioral data. 

The project covers automated data preprocessing, multi-model benchmarking, decision threshold calibration, artifact persistence, automated API test suites, a FastAPI prediction service, and a client-side interface.


## Problem Statement
Online retailers observe high traffic volumes, but only a small fraction of browsing sessions convert into transactions. Default classifiers operating at a static 0.50 cutoff often exhibit poor recall on high-intent sessions due to natural class imbalance. 

Predicting session conversion accurately allows platforms to trigger timely interventions without excessively badgering non-purchasing visitors.

---

## Objectives
1. Clean and transform session records using a leak-free pipeline architecture.
2. Train and evaluate linear and tree-based classification models under identical stratified splits.
3. Address conversion class imbalance via systematic classification threshold optimization.
4. Persist preprocessing steps and trained estimators as reusable inference artifacts.
5. Deploy a low-latency inference endpoint using FastAPI with strict Pydantic validation.
6. Provide an interactive frontend interface and robust test coverage using `pytest`.

---

## Dataset
**Name:** Online Shoppers Purchasing Intention Dataset  
**Records:** 12,330 sessions | **Features:** 18  
**Target:** `Revenue` (`0 = No Purchase`, `1 = Purchase`)

| Feature | Description |
|---|---|
| Administrative, Administrative_Duration | Number and duration of account/admin pages visited |
| Informational, Informational_Duration | Number and duration of informational pages visited |
| ProductRelated, ProductRelated_Duration | Number and duration of product/catalog pages visited |
| BounceRates | Average bounce rate of pages visited by the user |
| ExitRates | Average exit rate of pages visited by the user |
| PageValues | Average value of pages visited prior to transaction completion |
| SpecialDay | Closeness of browsing date to a specific holiday/special day |
| Month | Month of the browsing session |
| OperatingSystems, Browser, Region, TrafficType | System, browser, network, and demographic indicators |
| VisitorType | Returning Visitor, New Visitor, or Other |
| Weekend | Indicator for session occurring on Saturday or Sunday |
| **Revenue** | **Target** — Boolean indicator of transaction completion |

---

## Technologies Used
- **Machine Learning & Modeling:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Joblib
- **API & Serving:** FastAPI, Pydantic, Uvicorn
- **Frontend Interface:** HTML5, CSS3, JavaScript
- **Visualization:** Matplotlib
- **Testing & Environment:** Pytest, HTTPX, Git

---

## Architecture & Workflows

The system decouples offline model training from real-time API inference.

### 1. Offline Training Architecture
```mermaid
flowchart TD
    A[Online Shoppers Dataset] --> B[Data Ingestion & Cleaning]
    B --> C[Stratified Train/Test Split]
    C --> D[Preprocessing Pipeline: Median Imputer + OneHot]
    D --> E1[Logistic Regression]
    D --> E2[Random Forest]
    D --> E3[XGBoost]
    E1 & E2 & E3 --> F[Model Comparison Metrics]
    E3 --> G[Threshold Sweep: 0.20 to 0.70]
    G --> H[Select Operating Cutoff: 0.35]
    H --> I[Serialize Pipeline & Metadata: joblib]