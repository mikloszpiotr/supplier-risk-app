# Supplier Risk Assessment App

## Overview
This Streamlit app predicts risk levels for suppliers based on performance metrics.

## Structure
- `data/`
  - `raw/` – includes synthetic `suppliers.csv`
  - `processed/` – for cleaned data
- `models/` – trained model outputs
- `src/` – source modules
- `train.py` – script to train & save the classifier
- `app.py` – Streamlit dashboard
- `requirements.txt`

## Usage
1. Train model:
   ```
   python train.py
   ```
2. Run dashboard:
   ```
   streamlit run app.py
   ```
