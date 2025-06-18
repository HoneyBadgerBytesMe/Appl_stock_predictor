# Appl_stock_predictor
# Real-Time Local Stock Price Predictor

A self-contained, end-to-end machine learning pipeline that fetches live AAPL data, trains an XGBoost regression model on historical prices, and serves real-time predictions via a Flask API.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Tech Stack](#tech-stack)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
   - [1. Train & Serve API](#1-train--serve-api)  
   - [2. Fetch & Predict Loop](#2-fetch--predict-loop)  
6. [API Reference](#api-reference)  
7. [Development & Testing](#development--testing)  
8. [Project Structure](#project-structure)  
9. [Roadmap](#roadmap)  
10. [License](#license)  

---

## Project Overview

This project demonstrates an end-to-end ML system built locally—from data ingestion through model training to low-latency, on-demand inference. It:

- Downloads 10+ years of AAPL historical data and live minute-by-minute quotes via `yfinance`.  
- Engineers features (e.g., previous close, daily range, percent change).  
- Trains an `XGBRegressor` with tuned hyperparameters.  
- Exposes a Flask API (`/predict`) to receive JSON payloads and return predictions.  
- Automates live data fetching and prediction every 5 minutes.

---

## Key Features

- **Data Engineering**  
  - Historical + live data from Yahoo Finance  
  - Robust feature pipeline: Open/High/Low, Volume, Prev_Close, Daily_Range, Pct_Change  

- **Modeling**  
  - XGBoost regression with time-series validation  
  - Achieves **MSE &lt; 0.01**, **R² &gt; 0.9999**  

- **Deployment**  
  - Flask API serving predictions in &lt;200 ms  
  - Model persistence via `joblib`  

- **Automation**  
  - Background loop fetches live quotes & logs predictions every 5 minutes  

---

## Tech Stack

- **Language & Frameworks:** Python 3.8+, Flask  
- **Data:** `yfinance` for Yahoo Finance integration  
- **Modeling:** XGBoost (`xgboost`), scikit-learn  
- **Serialization:** joblib  
- **HTTP:** requests  

