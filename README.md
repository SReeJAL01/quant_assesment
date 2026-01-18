# NIFTY 5-Minute Quantitative Trading Pipeline

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)](https://github.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](https://opensource.org/licenses/MIT)

A comprehensive quantitative trading framework designed for the **NIFTY 50** index. This project implements an end-to-end pipeline covering data acquisition, advanced feature engineering, regime detection via Hidden Markov Models (HMM), EMA-based algorithmic strategies, and Machine Learning (XGBoost/LSTM) for signal enhancement.

---

## 📖 Table of Contents
* 🔭 Project Overview
* 🚀 Key Features
* 📂 Project Structure
* 🗒️Installation Instructions
* ❔How to Run
* 🔑 Key Results Summary

---

## 🔭 Project Overview

The objective of this project is to generate high-quality trading signals, backtest robust strategies, and extract actionable insights using a hybrid approach of technical indicators and options-based features.

### Core Workflow
1.  **Data Ingestion:** Fetch and process 5-minute OHLCV data for NIFTY Spot, Futures, and Options.
2.  **Feature Engineering:** Compute technical indicators (EMAs) and financial metrics (Options Greeks: Delta, Gamma, Vega, Theta, Rho).
3.  **Market Regime Detection:** Utilize Hidden Markov Models (HMM) to classify market states (e.g., Trending vs. Ranging).
4.  **Strategy Implementation:** Execute EMA crossover strategies filtered by market regimes.
5.  **ML Enhancement:** Train XGBoost and LSTM models to validate and filter trade signals.
6.  **Performance Analysis:** Deep dive into high-performing trades and outlier detection.

---

## 🚀 Key Features

* **Multi-Asset Data Processing:** Seamlessly cleans and merges Spot, Futures, and Options data.
* **Advanced Greeks Calculation:** Uses `mibian` to calculate real-time Options Greeks.
* **Regime-Based Filtering:** Dynamically adjusts strategy behavior based on HMM-detected market volatility.
* **Hybrid ML Models:** Combines traditional technical analysis with modern ML classifiers (XGBoost) and sequence models (LSTM).
* **Statistical Analysis:** Automated detection of 3-sigma outliers to identify high-impact market events.
---


## 📂 Project Structure

```bash
├── data/
│   └── All raw and processed CSV files
│
├── notebooks/
│   ├── main.py
│   ├── data_clean_merge.py
│   ├── feature_engineering.py
│   ├── regime_detection.py
│   ├── ema_backtest.py
│   └── mle.py
│
├── results/
│   └── Strategy performance metrics and analysis outputs
│
├── plots/
│   └── All generated visualizations and charts
│
└── README.md
```
---

## 🗒️ Installation Instructions

### Requirements
- Python 3.9+
- Packages:
```python
pip install pandas numpy matplotlib seaborn scipy hmmlearn xgboost tensorflow mibian
```
 
---


## ❔ How to Run

### Data Cleaning & Merging
- data/notebooks/main.py
- data/notebooks/data_clean_merge.py

### Feature Engineering
- data/notebooks/feature_engineering.py

### Regime Detection (HMM)
- data/notebooks/regime_detection.py

### EMA Backtest Strategy
- data/notebooks/ema_backtest.py

### ML-Enhanced Backtesting
- data/notebooks/mle.py

### High-Performance Trade Analysis
- Outputs are saved as CSVs and plots for visualization.

---

## 🔑 Key Results Summary

### EMA Backtest Strategy

- Total trades: ~1,398
- Win rate: ~51%
- Average trade duration: 2–4 candles
- Max drawdown: 1–2%

### ML-Enhanced Backtesting

- XGBoost and LSTM models improved trade filtering.
- Only trades predicted as profitable by ML were executed.
- Increased average PnL per trade while slightly reducing total trades.

### High-Performance Trade Analysis
- Outliers beyond 3-sigma identified (~1–2% of trades).

#### Key patterns:
- Most outliers occur in downtrend regime (-1).
- Outlier trades concentrated at market open hours.
- Average PnL of outliers significantly higher than normal profitable trades.
