# Evaluating Score-Investing Methodologies  
_A systematic review of Tweenvest’s algorithm for long-term stock investing using descriptive analytics and predictive modeling_

## 📌 Overview
This repository contains the code, datasets, and experiments developed as part of my **Integrated End of Majors Thesis** at the **Universitat Politècnica de València (UPV)**.  

The project investigates the **Tweenvest scoring algorithm**, which evaluates companies across four key factors: **Quality, Growth, Value, and Dividends**. The goal was to assess whether these scores can truly generate **alpha** (excess return over the market) across multiple investment horizons (1 month to 5 years).

By combining **financial theory, econometrics, data engineering, and machine learning**, the work provides insights into the effectiveness of factor-based investing and proposes methodological improvements.

---

## 🎯 Objectives
1. **System Architecture Enhancement**: Adapt Tweenvest’s infrastructure to store and retrieve historical score data.  
2. **Data Curation**: Preprocess, clean, and handle outliers in financial datasets.  
3. **Exploratory Analysis**: Analyze distributions, correlations, and patterns across factors.  
4. **Predictive Modeling**: Apply regression models, time series, and neural networks to evaluate predictive power.  
5. **Validation & Backtesting**: Benchmark models and check robustness across different horizons and regions.  

---

## 🛠️ Methodology
- **Data Preprocessing**: Consistency checks, transformations, and outlier detection (IQR, Isolation Forest, One-Class SVM, LOF).  
- **Predictive Models**:  
  - Linear & Generalized Additive Models (GAM)  
  - Time Series (ARIMA, windowed regressions)  
  - Neural Networks (binary classifiers, encoders, regressors)  
- **Backtesting**: Assessing score reliability across multiple timeframes and geographies.  

---

## 📂 Repository Structure
```

├── code/                        # Main Python package (data, models, utils, notebooks)
│   ├── data/                    # Processed datasets for analysis
│   ├── data_prep/               # Data preprocessing scripts
│   ├── descriptive_analysis.ipynb  # Exploratory analysis notebook
│   ├── general_regression/      # General regression models
│   ├── linear_regression/       # Linear regression models
│   ├── neural_network/          # Neural network models
│   ├── utils.py                 # Utility functions
│   └── test_mail.py             # Test scripts
├── scripts/                     # Standalone scripts (e.g., data splitting)
├── requirements.txt             # Python dependencies
├── LICENSE                      # License file
├── README.md                    # Project documentation (this file)
└── thesis.pdf                   # Full thesis document

```

---

## 🔧 Tech Stack
- **Languages**: Python (pandas, numpy, scikit-learn, statsmodels, PyTorch/TensorFlow)  
- **Data Engineering**: SQL, cron jobs, Redis for worker management  
- **Visualization**: matplotlib, seaborn  
- **Deployment**: Dedicated server setup with telemetry tracing  

---

## 📊 Key Findings
- Tweenvest’s factor scores show **partial predictive power**, but results vary across:  
  - **Time horizons** (short-term vs. long-term)  
  - **Geographical regions**  
  - **Score combinations**  
- Non-linearities and complex interactions exist, suggesting opportunities for model refinement.  
- Neural networks achieved **higher consistency** in certain cases, but interpretability remains a challenge.  

---

## 🚀 Future Work
- Incorporate **social listening** and qualitative data (news, transcripts, reviews).  
- Explore **multimodal AI approaches** for better alpha prediction.  
- Enhance backtesting with live market integration.  
- Apply explainable AI (XAI) for better interpretability of investment models.  

---

## 📖 Citation
If you use this work, please cite:  

**Carlos Eduardo Domínguez Martínez (2025).**  
*Evaluating Score-Investing Methodologies: A Systematic Review of Tweenvest’s Algorithm for Long-Term Stock Investing Using Descriptive Analytics and Predictive Modeling.*  
Universitat Politècnica de València.  

---

## 👤 Author
**Carlos Eduardo Domínguez Martínez**  
- 💼 Data Analyst | ML & Economy Enthusiast  
- 📧 **Email:** [contact@carloseduardo.es](mailto:contact@carloseduardo.es)
- 🌐 **Website:** [carloseduardo.es](https://carloseduardo.es)

---

