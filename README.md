# Walmart Store & Product Sales Forecasting Dashboard 🛒

An interactive, machine-learning-powered business intelligence dashboard and forecasting system designed to predict weekly sales for Walmart stores and product categories. This application provides real-time historical insights, predictive forecasting (up to 3 weeks out), and a model performance registry.

Developed with a custom dark Material You aesthetic, the application leverages time-series feature engineering and multiple scikit-learn regressor models.

---

## 🌟 Key Features

### 1. **Interactive Store Overview**
* **KPI Cards**: Track total historical sales, average transaction value, average customer ratings, and average profit margins.
* **Holiday Alerts**: Automated warnings for critical retail holidays (Thanksgiving, Christmas, New Year's, etc.) which typically experience high volatility.
* **Visual Analytics**: Interactive Altair charts demonstrating weekly sales trends, transaction counts, and category contribution.

### 2. **Advanced Forecasting Terminal**
* **Multi-Week Forecasts**: Generates predictions for the next 1st, 2nd, and 3rd weeks of sales for each product category within a selected store.
* **Intelligent Fallback**: If a machine learning model is missing or training data is insufficient, the system gracefully falls back to recent historical moving averages.
* **Trend Indicators**: Displays percentage changes relative to the prior week with color-coded positive, negative, or flat indicators.

### 3. **Model Registry & Performance Insights**
* **Model Selection Transparency**: Shows the exact algorithm selected as the "Best Model" for each category/store.
* **Performance Metrics**: Access model evaluation metrics including $R^2$ scores, Adjusted $R^2$, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

---

## 🧠 Machine Learning Pipeline & Architecture

The forecasting architecture uses time-series feature engineering paired with regression modeling.

### **Feature Engineering (`walmart_utils.py`)**
To capture temporal dependency and trends, the raw sales transaction data is resampled to a weekly frequency (`W`) and enriched with:
* **Lag Variables**: Sales from 1 week (`lag_1`), 2 weeks (`lag_2`), and 4 weeks (`lag_4`) prior.
* **Rolling Averages**: 4-week moving average of past sales (`ma_4`).
* **Calendar Features**: Iso-calendar week of the year, month, quarter, and year.
* **Aggregations**: Average unit price, average customer ratings, average profit margins, weekly transaction counts, and quantity sold.

### **Model Training Process**
Two scripts are provided for training:
1. **Category & Product Level (`train_store_models.py`)**:
   * Evaluates several candidates: **Linear Regression, Ridge, Lasso, Elastic Net, Polynomial Regression, Random Forest, and Gradient Boosting**.
   * Employs `GridSearchCV` with a `TimeSeriesSplit` cross-validation strategy.
   * **Hierarchical Optimization**:
     1. First tries to fit a category-level weekly forecasting model.
     2. If the $R^2$ score is below a target threshold of **0.90**, it attempts store-specific models for that category.
     3. If that fails, it builds a branch-aware weekly model using store dummy variables.
     4. If all else fails, it trains a branch-aware daily model to find the best possible fit.
     5. Saves the selected model as a `.joblib` file in the `models/` directory and writes the results to `training_summary.csv`.
2. **Store & Branch Level (`train_store_models_storewise.py`)**:
   * Trains store-wise regression models for individual stores (e.g. store codes `M001`, `M002`, etc.) and outputs results to `training_summary_storewise.csv`.

---

## 📂 Project Directory Structure

```directory
Walmart/
├── .streamlit/
│   └── config.toml               # Streamlit page styling & dark-theme configurations
├── models/                       # Directory containing trained joblib models (.joblib)
├── app.py                        # Main Streamlit dashboard application
├── data.csv                      # Historical Walmart transaction dataset
├── train_store_models.py         # Model training script for product categories
├── train_store_models_storewise.py # Model training script for individual stores
├── walmart_utils.py              # Data cleaning, resampling, and feature engineering utilities
├── training_summary.csv          # Evaluation metrics registry for category models
├── training_summary_storewise.csv# Evaluation metrics registry for store models
├── requirements.txt              # Project package dependencies
└── README.md                     # Project documentation
```

---

## 🛠️ Installation & Setup

### **1. Prerequisites**
* Python 3.8 or higher installed on your system.

### **2. Setup Environment & Install Dependencies**
Clone or download the project folder, open a terminal in the project directory, and run:

```bash
# Create a virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### **3. Train the Models**
Before launching the dashboard, make sure to train the forecasting models:

```bash
# Train product category forecasting models (looks for threshold R2 >= 0.90)
python train_store_models.py

# Train store-specific forecasting models
python train_store_models_storewise.py
```
*Note: This will populate the `models/` directory with `.joblib` bundles containing models, feature columns, and category info.*

### **4. Run the Streamlit Dashboard**
Run the Streamlit server locally:

```bash
streamlit run app.py
```
This will automatically launch the dashboard in your default browser (usually at `http://localhost:8501`).

---

## 📦 Requirements
* `streamlit`
* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`
* `altair`

---

## 🎨 Design & Aesthetic Customization
The dashboard is styled using modern CSS rules matching a custom dark theme (`#0B0F19` background, `#171C28` card colors, `#3B82F6` accents, and `Plus Jakarta Sans` typography). Streamlit configurations are synced via `.streamlit/config.toml`.

---

## The Website URL 
https://walmart-weekly-sales-forecasting.streamlit.app
