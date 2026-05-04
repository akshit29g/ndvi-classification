# NDVI Classification (Satellite Data)

## Overview
Built a model to classify land cover types using NDVI time-series data with noise due to cloud interference.

## Tech Stack
Python, LightGBM, Optuna, Pandas, Scikit-learn

## Approach
- Data cleaning and imputation
- Feature engineering (NDVI, band ratios)
- Model training using LightGBM
- Hyperparameter tuning using Optuna

## Result
Top 15% rank in Kaggle competition (1300+ teams)

## Run
pip install -r requirements.txt  
python train.py