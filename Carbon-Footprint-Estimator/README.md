# 🌍 Carbon Footprint Estimator

> Predict annual carbon emissions from lifestyle, transport, and geospatial data using Machine Learning regression models.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![ML](https://img.shields.io/badge/ML-XGBoost-green)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)

---

## 📁 Project Structure

```
Carbon-Footprint-Estimator/
├── data/
│   ├── raw/                          # Generated raw datasets
│   │   ├── lifestyle_transport.csv   # Lifestyle & transport survey data
│   │   └── geo_profiles.csv          # Country geospatial profiles
│   └── processed/                    # Cleaned data & plots
│       └── features_final.csv
├── notebooks/
│   ├── 01_EDA.py                     # Exploratory Data Analysis
│   └── 02_Model_Evaluation.py        # Model comparison & SHAP
├── src/
│   ├── utils.py                      # Config, logging, constants
│   ├── data_loader.py                # Load & validate data
│   ├── geo_features.py               # Geospatial feature engineering
│   ├── preprocessing.py              # Cleaning, encoding, scaling
│   ├── model.py                      # Train & compare models
│   └── predict.py                    # Inference pipeline
├── app/
│   └── app.py                        # Streamlit web application
├── models/
│   ├── best_model.pkl                # Saved best model
│   └── preprocessing_artifacts.pkl   # Saved encoders & scaler
├── generate_data.py                  # Synthetic data generator
├── config.yaml                       # Configuration & hyperparameters
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
```bash
python generate_data.py
```

### 3. Train Models
```bash
python src/model.py
```

### 4. Run the Web App
```bash
streamlit run app/app.py
```

### 5. Command-Line Prediction
```bash
python src/predict.py --country India --diet Omnivore --electricity 300 --heating "Natural Gas" --members 4 --vehicle Petrol --commute 15 --flights_short 2 --flights_long 1
```

## 🧪 Notebooks

Run in VS Code (interactive Python cells with `# %%`):

- **01_EDA.py** — Data exploration, distributions, correlation heatmap, geographic maps
- **02_Model_Evaluation.py** — Model comparison charts, residuals, feature importance, SHAP

## 🤖 Models Compared

| Model | Description |
|---|---|
| Ridge Regression | Baseline linear model |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting |
| XGBoost | Optimized gradient boosting (usually best) |

## 🌐 Geospatial Features

The key differentiator — location-based features that capture regional differences:

- **Grid Carbon Intensity** — CO₂ per kWh of local electricity
- **Climate Zone** — Tropical, Arid, Temperate, Continental, Polar
- **Urban Density** — Population per km²
- **Heating/Cooling Demand Index** — Climate-driven energy needs

## 📊 Key Technologies

- **Data**: Pandas, NumPy
- **ML**: Scikit-learn, XGBoost, LightGBM
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Explainability**: SHAP
- **Deployment**: Streamlit
- **Geospatial**: GeoPandas concepts

## 📝 License

MIT License — free to use and modify.
