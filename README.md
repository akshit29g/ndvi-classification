# NDVI-based Land Cover Classification

## Overview
This project focuses on classifying land cover types using NDVI (Normalized Difference Vegetation Index) time-series data derived from satellite imagery. The dataset contains significant noise due to cloud interference and labeling inaccuracies, making preprocessing and feature engineering critical for achieving good performance.

---

## Problem Statement
- Classify land cover into categories such as Water, Forest, Grass, Farm, etc.
- Handle noisy NDVI signals caused by cloud cover
- Manage missing values in time-series data
- Ensure model generalizes well on clean unseen data

---

## Dataset Description
- Each sample contains NDVI values recorded at multiple timestamps
- Represents temporal vegetation patterns
- Includes:
  - Noisy training data
  - Mixed noisy + clean test data
- Target variable: land cover class

> Dataset provided as part of Summer Analytics Hackathon (not included due to size and usage constraints)

---

## Approach

### 1. Data Preprocessing
- Removed irrelevant columns (ID, unnamed fields)
- Handled missing values using median imputation

### 2. Feature Engineering
- Computed NDVI from spectral bands (NIR, Red)
- Created spectral band ratios (Blue/Green)
- Extracted time-series statistical features:
  - Mean NDVI
  - Standard deviation
  - Maximum and minimum values

### 3. Model Selection
- Used LightGBM for efficient handling of tabular data
- Captures non-linear relationships and feature interactions

### 4. Hyperparameter Optimization
- Applied Optuna for efficient hyperparameter tuning
- Used validation split with early stopping to avoid overfitting

---

## Results
- Achieved **Top 15% rank out of 1300+ teams**
- Demonstrated strong generalization despite noisy training data

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- LightGBM
- Optuna

---

## Project Structure
ndvi-classification/
├── src/
│ └── train.py
├── README.md
├── requirements.txt
└── .gitignore


---

## How to Run

1. Install dependencies:
pip install -r requirements.txt

2. Place dataset files in root directory:
- train.csv  
- test.csv  

3. Run the project:
python src/train.py

---

## Key Learnings
- Handling noisy real-world satellite data
- Importance of feature engineering in tabular machine learning
- Efficient hyperparameter tuning using Optuna
- Building a complete end-to-end ML pipeline

---

## Future Improvements
- Use cross-validation instead of a single validation split
- Experiment with ensemble methods for better performance
- Explore deep learning models for time-series data (LSTM, Transformers)
- Incorporate spatial/image-based features if available
