# Stock-price-movement-prediction-model
This project is a comprehensive machine learning pipeline designed to predict short-term stock price movements using technical indicators, price-volume features, breakout patterns, and volatility measures.
It forecasts the average 5-day forward return based on past market data, providing insights into which technical features most influence price movement.

The methodology is inspired by quantitative finance research and aims to bridge the gap between financial domain knowledge and data science techniques.
# Features & Highlights
-Data Preprocessing:
---Handles Excel-specific formatting issues
---Cleans missing values & converts data types

-Target Engineering:
---Computes forward 5-day return as prediction target
---Labels data for supervised learning

-Feature Engineering:
---Price-based metrics: Closing strength, wick %, HiLo %
---Volume-based metrics: Volume z-score, delivery ratio
---Volatility measures: ATR ratio, Bollinger Band width %
---Breakout detection: NR7 pattern, volume spikes

-Exploratory Data Analysis (EDA):
---Feature group correlations
---Multi-level factor analysis

-Modeling:
---Linear Regression with outlier handling
---Train-test split performance metrics

-Insights:
---Top predictive features identified
---Feature group importance analysis

# Tech Stack
-Programming Language: Python 3.x
-Libraries:
---pandas, numpy
---matplotlib, seaborn
---pandas_ta
---scikit-learn
