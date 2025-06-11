# 📊 Supplier Risk Assessment

A Streamlit dashboard that automatically scores and visualizes the risk levels of your suppliers—helping procurement teams flag trouble early, reduce disruptions, and cut audit costs.

---

## 🏭 Business Problem

Large manufacturing and retail organizations rely on dozens—or hundreds—of suppliers—for raw materials, sub-assemblies, and finished goods. Each supplier carries operational, financial, and reputational risk:

- **Delivery delays** can stall production lines or retail shelves.  
- **Quality failures** lead to rework, scrap, or customer complaints.  
- **Financial instability** (e.g., late payments, cash-flow issues) may signal potential shutdowns.  
- **Reputational threats** (from negative news sentiment) can endanger brand image.

**Without an automated risk-scoring system**, procurement teams must manually comb through spreadsheets and disparate systems—an error-prone, reactive process.

---

## 💡 Solution Overview

This project uses an **XGBoost** classifier trained on historical supplier data to predict **High / Medium / Low** risk. It then presents:

- **Risk Distribution**: How many suppliers fall into each bucket  
- **Feature Importances**: Which factors (delivery rate, quality defects, financial health, news sentiment) drive risk most  
- **Trend Over Time**: Monthly evolution of risk counts  
- **Summary Metrics**: Clear KPI cards for executives

By embedding this into a Streamlit app, stakeholders get an **interactive, self-service dashboard** that flags high-risk partners before they cause disruptions.

---

## 📂 Repository Structure

```
supplier-risk-app/
├── data/
│   ├── raw/                # raw CSV (e.g. suppliers.csv with a risk_label)
│   └── processed/          # (optional) cleaned, feature-engineered data
├── models/
│   ├── classifier.pkl      # trained XGBoost model
│   └── label_encoder.pkl   # label encoder for risk classes
├── src/                    # core Python modules
│   ├── preprocessing.py    # load, clean & feature engineer
│   ├── modeling.py         # train_model() and save_model()
│   └── predict.py          # predict_risk() and feature_importances()
├── app.py                  # Streamlit dashboard entrypoint
├── train.py                # one-shot script to train & save the model
├── requirements.txt        # Python dependencies
└── README.md               # (this) overview & instructions
```

---

## 🚀 Quickstart

1. **Clone the repo**  
   ```bash
   git clone https://github.com/mikloszpiotr/supplier-risk-app.git
   cd supplier-risk-app
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**  
   Ensure `data/raw/suppliers.csv` exists (with columns:  
   `supplier_id`, `on_time_deliveries`, `total_deliveries`, `quality_failures`,  
   `financial_health_score`, `sentiment_score`, `order_date`, `risk_label`).

   ```bash
   python train.py
   ```

4. **Run the dashboard**  
   ```bash
   streamlit run app.py
   ```
   Open [http://localhost:8501](http://localhost:8501) to explore.

---

## 🌐 Deploy to Streamlit Cloud

1. Push your code to GitHub on `main`.  
2. Log in at [streamlit.io/cloud](https://streamlit.io/cloud).  
3. Create a “New app”:
   - **Repository**: `mikloszpiotr/supplier-risk-app`  
   - **Branch**: `main`  
   - **File path**: `app.py`  
4. Click **Deploy**.  
5. Share your live URL (e.g.  
   `https://supplier-risk-app-mikloszpiotr.streamlit.app`).

---

## 📊 How It Works (ML Details)

1. **Feature Engineering**  
   - **On-time delivery rate** = on_time_deliveries / total_deliveries  
   - Raw counts of quality failures  
   - Financial health score (normalized)  
   - News sentiment score (–1 to +1)  

2. **Model Training**  
   - **Algorithm**: XGBoost classifier  
   - **Hyperparameters**: 100 trees, learning rate 0.1, random state 42  
   - **Target**: `risk_label` (“High”, “Medium”, “Low”) encoded via `LabelEncoder`

3. **Prediction & Visualization**  
   - `predict_risk()` returns class labels  
   - `model.feature_importances_` drives the feature chart  
   - Results power the Streamlit charts and KPI cards

---

## ⚙️ Customization

- Swap in your own dataset: replace `data/raw/suppliers.csv`.  
- Tweak feature definitions in `src/preprocessing.py`.  
- Experiment with other models (LightGBM, RandomForest) by editing `src/modeling.py`.  
- Add external data sources (e.g. credit-rating APIs) to enrich features.

---

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
