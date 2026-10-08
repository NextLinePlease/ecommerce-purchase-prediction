# 🛒 E-Commerce Purchase Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![Scikit Learn](https://img.shields.io/badge/ML-Scikit--Learn-red)

An end-to-end machine learning project that predicts whether an online shopping session will result in a purchase.

The project includes data preprocessing, exploratory data analysis, machine learning, model evaluation, and a REST API for real-time predictions.

---

## 🎯 Problem Statement

E-commerce websites generate large amounts of user browsing data.

The goal of this project is to predict whether a visitor will complete a purchase based on their browsing behavior.

The model takes session-level information such as:

- Number of product pages visited
- Session duration
- Bounce rate
- Exit rate
- Page value
- Visitor type
- Traffic type
- Browser
- Operating system
- Region
- Weekend activity

and predicts the probability of purchase.

---

## 🚀 Project Features

- Data preprocessing
- Numerical feature preprocessing
- Categorical feature encoding
- Exploratory Data Analysis
- XGBoost classification
- Model evaluation
- Purchase probability prediction
- FastAPI REST API
- Interactive Swagger API documentation
- Saved trained model using Joblib

---

## 📊 Dataset

### Online Shoppers Purchasing Intention Dataset

The dataset contains **12,330 online shopping sessions** and **18 features**.

The target variable is:

```text
Revenue