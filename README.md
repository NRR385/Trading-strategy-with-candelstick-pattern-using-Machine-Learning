# Trading-strategy-with-candelstick-pattern-using-Machine-Learning
AI-Driven Trading System using Ridge Regression for 3-day recursive price forecasting. It features VROC (Volume Rate of Change) to filter market noise and identifying false signals. Built with Python &amp; Streamlit, it delivers real-time interactive charts, currency conversion, and ~57% directional accuracy for smarter, data-driven decisions.
# 📈 AI-Driven Financial Trading & Forecasting System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> **A Real-Time Financial Dashboard that uses Recursive Machine Learning (Ridge Regression) to predict stock prices for the next 3 days.**

---

## 📖 Table of Contents
- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Installation & Setup](#-installation--setup)
- [How It Works](#-how-it-works)
- [Future Scope](#-future-scope)
- [Contact](#-contact)

---

## 📌 About The Project

Retail traders often struggle with emotional decision-making and reliance on lagging indicators (like simple Moving Averages) that only reflect past market behavior. Existing tools are either too simplistic or expensive "black-box" systems.

**This project solves that gap.**

It is a **Hybrid Analytical System** that combines:
1.  **Technical Analysis:** Uses proven indicators like **RSI**, **EMA**, and **VROC (Volume Rate of Change)**.
2.  **Machine Learning:** Implements **Ridge Regression** with L2 Regularization to handle market noise.
3.  **Recursive Forecasting:** Instead of just predicting tomorrow, it predicts **T+1**, **T+2**, and **T+3** days ahead to show the trend direction.

---

## 🚀 Key Features

* **🔮 Recursive Multi-Step Forecasting:** Predicts stock prices for the next 3 days (72 hours) using a recursive loop strategy.
* **📊 Interactive Real-Time Charts:** Features professional-grade, zoomable candlestick charts powered by `lightweight-charts`.
* **🧠 Noise Reduction AI:** Uses Ridge Regression to filter out multicollinearity and prevent overfitting on volatile data.
* **📉 Smart VROC Filtering:** Validates price breakouts by analyzing Volume Rate of Change—ignoring "fake" pumps.
* **🌍 Universal Asset Support:** Works with **Stocks** (Reliance, Apple), **Crypto** (Bitcoin, Ethereum), and **Forex** via Yahoo Finance.
* **💱 Currency Converter:** Automatically converts USD assets to **INR (₹)** for Indian traders.

---

## 🛠 Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Language** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core Logic & Scripting |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) | Web Dashboard UI |
| **ML Core** | ![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Ridge Regression Model |
| **Data** | ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data Manipulation & Time-Series |
| **API** | **yfinance** | Live OHLCV Market Data |

---

## 🏗 System Architecture

The system follows a 4-Layer Architecture:

1.  **Data Layer:** Fetches live data from Yahoo Finance API.
2.  **Processing Layer:** Cleans data, fixes Timezones (UTC -> IST), and calculates Technical Indicators.
3.  **Intelligence Layer:** Runs the Ridge Regression model to generate the 3-Day Forecast.
4.  **Presentation Layer:** Displays the Dashboard and Signals via Streamlit.

*(You can upload your architecture diagram image here)*

---

## ⚡ Installation & Setup

Follow these steps to run the project locally.

### Prerequisites
* Python 3.8 or higher installed.

### Steps
1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/AI-Trading-System.git](https://github.com/your-username/AI-Trading-System.git)
    cd AI-Trading-System
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **Access the Dashboard**
    Open your browser and go to `http://localhost:8501`.

---

## 🧠 How It Works
1.  **Select Asset:** Enter a ticker symbol (e.g., `BTC-USD` or `RELIANCE.NS`).
2.  **Data Fetch:** The system pulls the last 60 days of OHLCV data.
3.  **Feature Engineering:** It calculates RSI, EMA, and VROC.
4.  **Training:** The Ridge Model trains on this data instantly.
5.  **Forecasting:**
    * Predicts Price for Day 1.
    * Appends Day 1 prediction to data -> Predicts Day 2.
    * Appends Day 2 prediction to data -> Predicts Day 3.
6.  **Visualization:** Plots the historical data and the future 3-day trend line.

---

## 🔮 Future Scope
* **Sentiment Analysis:** Integrating News API to analyze market sentiment.
* **Deep Learning:** Testing LSTM models for long-term trend analysis.
* **Live Alerts:** Sending Email/WhatsApp alerts when a Buy signal is detected.

---

## 🤝 Contact
**Name:** [Your Name]
**Email:** [Your Email]
**Project Link:** [https://github.com/your-username/AI-Trading-System](https://github.com/your-username/AI-Trading-System)
