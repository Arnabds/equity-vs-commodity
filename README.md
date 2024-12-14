# MarketMind : Unveiling Stock Trends with Machine Learning
Erdos Data Science Bootcamp - Fall 2024

A foundational step in predicting the stock market of a company by leveraging stock market prices of other companies and related commodities.

<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Motivation">Motivation</a></li>
    <li><a href="#Pre-analysis">Pre-analysis</a>
    </li>
    <li><a href="#Data">Data</a>
        <ul>
            <li><a href="#Data-Cleaning-and-Preparation">Data Cleaning and Preparation</a></li>
            <li><a href="#Financial-Factors">Financial Factors</a></li>
        </ul>
    </li>
    <li><a href="#EDA-and-Preprocessing">EDA and Preprocessing</a>
    <li><a href="#Models">Models</a>
        <ul>
            <li><a href="#Continuous-Models">Continuous Models</a></li>
            <li><a href="#Classification-Models">Classification Models</a></li>
        </ul>
    </li>
    <li><a href="#Investopedia-Simulation">Investopedia Simulation</a></li>
    <li><a href="#Model-Performance-Comparison">Model Performance Comparison</a></li>
    <li><a href="#Summary">Summary</a></li>
    <li><a href="#Things-to-answer-and-to-be-updated-next">Things to Answer and Update Next</a></li>
    <li><a href="#Code-description">Code Description</a></li>
</ul>

---

<h3 id="Motivation">Motivation</h3>

The stock market is complex - volatile, multifactorial, and often lacking actionable data. Our mission was to address this challenge through rigorous analysis, innovative feature engineering, and robust forecasting techniques. Our study is centered on technical indicators rather than developing an algorithmic trading strategy. Our objective is to evaluate whether machine learning models can predict future stock price movements. While our approach does not directly aim to maximize trading profits, it offers valuable insights that traders and analysts can incorporate into broader strategic decisions.

---

<h3 id="Pre-analysis">Pre-analysis</h3>

We began by studying the time series of gold commodity prices. Using data imported from Yahoo Finance, we performed time series analyses. The available data included `Volume`, `High`, `Low`, `Open`, and `Close` prices, but we focused solely on `Close` prices for analysis. 

<img src=/images/ARIMAresidual.png width="800" class="center" />

Our pre-analysis revealed that while ARIMA achieved theoretical stationarity, residual volatility persisted, even beyond GARCH's control. This underscores the need for additional predictive variables beyond historical prices to better understand market behavior.

<img src=/images/GARCH(1,1).png width="800" class="center" />

---
<h2 id="Data">Data</h2>

Our focus is on Ford (`'F'`), a major player in the U.S. automotive sector, as the testbed for this analysis. We consider 26 other companies that are directly or indirectly connected to Ford, including but not limited to companies in the same field and its supplier companies.

- **Automotive companies:** `'GM'` (General Motors Co.), `'TM'` (Toyota Motor Corp)
- **Supplier companies:** `'ALV'` (Autoliv Inc), `'DNZOY'` (Denso ADR), `'ASEKY'` (Aisin Corp), `'JCI'` (Johnson Controls International PLC)
- **Commodities:** `'CL=F'` (Crude Oil), `'X'` (US Steel Corporation), `'ALI=F'` (Aluminum)
- **Currencies exchange rates:** `'EURUSD=X'` (EUR/ USD), `'CHFUSD=X'` (CHF/ USD), `'GBPUSD=X'` (GBP/ USD), `'JPYUSD=X'` (JPY/ USD), `'CADUSD=X'` (CAD/ USD), `'INRUSD=X'` (INR/ USD), `'RUBUSD=X'` (RUB/ USD). `'TRYUSD=X'` (TRY/ USD)
- **Financial companies and IRX:** `'^IRX'` (Inhalerx Ltd), `'JPM'` (JPMorgan Chase & Co), `'BAC'` (Bank of America Corp),`'C'` (Citigroup Inc), `'WFC'` (Wells Fargo & Co)
- **Others:** `'^IXIC'` (Nasdaq Composite), `'^GSPC'` (S&P), `'^DJI'` (Dow Jones Industrial Average), `'FDX'` (FedEx Corp)

We also included currency exchange rates in the dataset.

To refine our feature selection, we:

    1. Drew a correlation matrix of closing prices.
    2. Selected stocks with an absolute correlation > 0.4.
    3. Applied Granger Causality to further narrow the list.

<p float="center">
  <img src="/images/Ford_connections.png" width="600" /> 
</p>
--- 

<h3 id="Data-Cleaning-and-Preparation">Data Cleaning and Preparation</h3>


Granger Causality is a statistical hypothesis test used to determine whether one time series can predict another. In other words, if the past values of one time series `X` provide significant information about the future values of another variable `Y` (beyond what is contained in the past values of `Y` alone), the `X` is said to Granger-cause `Y`. Granger Causality is closely related to cross-correlation, it is more sophicated since it provides a test for predictive causality, an information that can be useful for statistical modeling.

We found that 21 of 26 companies have strong correlations to Ford. Applying the Granger-causality test, we reduced our relevant companies to 6. The final set of features included: `'CADUSD=X'`, `'GM'`, `'JCI'`, `'TM'`, `'TRYUSD=X'`, `'^IXIC'`, and `'F'`. 

We obtained raw data from `yfinance` and cleaned data between **November 26, 2019** and **October 28, 2024**:

[dataset_others](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Data/dataset_others.csv): data for regression models.

[dataset_others_class](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Data/dataset_others_class.csv): indicators data for classification model. 

<img src=/images/data.png width="800" class="center" />

We assessed model performance for continuous and categorical prediction using continuous model such as LSTM, and Classification model like Logistic Regression. We employed forward cross-validation and backtesting to evaluate model robustness. The validation set was `F`'s trading data between **November 4, 2024** and **November 29, 2024**. We splitted the data set into training set and backtesting set. Backtesting set was not used in the training process, but it was reserved for the final strategy backtest.

<img src=/images/preprocessing.jpg width="800" class="center" />

--- 

<h3 id="Financial-Factors">Financial Factors</h3>

Financial factors introduced missing observations (e.g., from rolling calculations like moving averages). We removed these rows to ensure a clean dataset. This prepared dataset was used for regression, logistic regression, and advanced machine learning models, including random forests, gradient boosting, SVMs, and neural networks. The cleaned data was saved for further modeling.

Given the daily granularity of Yahoo Finance data, the predictors derived from the raw data were insufficiently meaningful due to multicollinearity. To address this, we engineered the following financial factors:

- **Gain** is an increase in stock's market value from its purchase price.
- **Average Gain** is average gain over a period of 14 days.
- **Loss** is a decrease in stock's market value from its purchase price. 
- **Average Loss** is average loss over a period of 14 days.
- **Relative Strength Index (RSI)** determines whether the stock is overbought or oversold.
- **Moving Averages** identifies trend direction of a stock.
- **Simple Moving Average** calculates rolling average of close prices over a period of 20 days. 
- **Exponential Moving Average** calculates exponential average of close prices over a period 20 days and weights average that gives greater importance to the price in more recent days. 
- **Rate of Change** measures the most recent change in price with respect to the price 12 days ago.
- **Price Volume Trend** determines a security's price direction and strength of price change.

---

<h2 id="EDA-and-Preprocessing">EDA and Preprocessing</h2>

![Overall model with bollinger band](/images/Overall_model_with_bollinger_band.png)


This is the correlation plot with all the final selected features.
![Correlation plot](/images/corr.png)

---
<h2 id="Models">Models</h2>

- This project has 3 primary goals:
  - Train a model that accurately predicts the closing price value.
  - Train a model that accurately classify whether the stock closing price will go up or down the next day.
  - Invest fake money in investopedia to check our model's prediction performance.
- We must use an appropriate validation scheme to select the best model!

<h3 id="Continuous-Models">Continuous Models</h3>

---

<h4>Regression and XGBoost Models</h4>
We have fitted 4 models from linear regression, linear regression with PCA, linear logistic regression, linear logistic regression with PCA all with regularization, XGBoost, XGBoost PCA.

<p float="left">
  <img src="/images/Reg.png" width="400" /> 
  <img src="/images/Reg_pca.png" width="400" /> 
  <img src="/images/XGboost_reg.png" width="400" />
  <img src="/images/XGBoost_reg_pca.png" width="400" /> 
</p>

This table summarizes the **average accuracy** achieved across five backtesting iterations using ridge, lasso, and elastic net regularization, with optimized hyperparameters. The models consistently demonstrated effective directional predictions and reliable performance, even in volatile stock market scenarios.

| **Models**               | **Subparts**                                                   | **Accuracy**        | **Average Accuracy** |
|--------------------------|----------------------------------------------------------------|---------------------|-----------------------|
| Regression               | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 85.6%      | **85.6%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 42.9%     | **82.9%**                        |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 100%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 71.4%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 100%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 100%     |                       |
| Regression PCA           | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 84%      | **84%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 71.4%     | **82.8%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 85.7%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 71.4%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 100%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 85.7%     |                       |
| XGBoost                  | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 81%      | **81%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 42.9%     | **82.9%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 100%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 85.7%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 85.7%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 100%     |                       |
| XGBoost PCA              | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 83%      | **83%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 86.9%     | **86.6%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 89.8%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 85.5%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 85.5%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 85.5%     |                       |
| Logistic Regression      | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 66.7%      | **66.7%**            |
| Logistic Regression PCA  | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 78%      | **78%**            |
---

<h4>Long short-term memory (LSTM)</h4>

Recurrent Neural Networks (RNNs) is well-suited for time series modeling.
Among RNNs, Long Short-Term Memory modules are especially advantageous for capturing temporal dependencies.
Unlike many other classical models, this approach eliminates the need for preselecting input features, simplifying the process.
During the training phase, historical data from highly correlated tickers organized based on Granger causality, and their various indicator functions, is used as input.
The input data is processed through a stacked LSTM module and a fully connected layer in turn.
Finally, the model produces a continuous output value representing the predicted stock price of interest.

![LSTM](/images/LSTM.JPG)

The trained network was applied to real-time market data for daily predictions during the last November.
The model performed well, achieving an accuracy of 78.9%, correctly predicting 15 out of 19 cases.

<p float="left">
  <img src="/images/LSTM_close.png" width="400" />
  <img src="/images/LSTM_class.png" width="400" /> 
</p>

---

<h4>Support Vector Regression</h4>

We used Supprt Vector Regression to predict Ford's stock movement. We selected models from time series cross validation and grid search. We found the best hyperparameters are {C: 1, epsilon: 0.01, kernel: 'linear'} and yield MSE of 0.0081. The confidence interval showed high uncertainty in forecasting future prices. It struggled to predict the exact magnitude of the changes, such as the steepness of the slope or the precise amount of up or down movement.

![SVM_regressor](/images/SVM_regression.JPG)

---

<h4>Vector Auto-Regression (VAR)</h4>

If X Granger-causes Y, then we can use X to predict Y. From the Granger Causality test, we found that `'CADUSD=X'`, `'GM'`, `'JCI'`, `'TM'`, `'TRYUSD=X'`, `'^IXIC'`  Granger-cause `'F'`, therefore including information of these companies will improve prediction. We implemented the Vector AutoRegressive on training set to determine `lag order` for each company. The `lag order` refers to the number of past days that are included in the model.

The table was obtained across 5-fold cross-validation, where `n` is forecast length, so `n = 5` is the prediction for one week of trading. The `optimal lag` ranges from 6 days to 39 days. VAR models of
`CADUSD=X` had small `lag order` while `JCI` required more past observations compared to the rests. 

| Ticker              | n = 1  | n = 2 | n = 3 | n = 4 | n = 5 |
|---------------------|--------|-------|-------|-------|-------|
| CAD/USD             | 21     | 7     | 6     | 8     | 7     |
| GM                  | 14     | 12    | 29    | 29    | 16    |
| JCI                 | 33     | 30    | 33    | 22    | 22    |
| TM                  | 27     | 30    | 26    | 27    | 27    |
| TRY/ USD            | 22     | 22    | 18    | 39    | 39    |
| IXIC                | 27     | 26    | 26    | 26    | 26    |

We applied our models to 10 backtesting sets. Toyota Motors (`TM`) achieved accuracy of 70% for shorter forecast length (`n = 1' and `n = 2`), while General Motors (`GM`) performed better on longer forecast length (`n = 3` and `n = 4`). If we wanted to predict one week of trading (5 trading days), Nasdaq (`IXIC`) could get up to 62% accuracy.  

<p float="center">
  <img src="/images/VAR_backtest.jpg" width="800" />
</p>

We also used VAR models to predict `F`'s close values between **November 20, 2024** and **November 29, 2024** (7 trading days). The confidence interval was wide, indicating that the sample did not provide a precise representation of the mean, even though the model could predict the stock trends.

<p float="center">
  <img src="/images/VAR_close_price.png" width="800" /> 
</p>

---

<h3 id="Classification-Models">Classification Models</h3>

---

<h4>KNN</h4>

We apply KNN with different metrics (e.g., Euclidean, Manhattan) using time-series split. We train on smaller data (last 2 month) and include Dynamic Time Warping (DTW), which is better for time-series but computationally intensive. Accuracies indicate KNN often performs close to random guessing, showing poor alignment with time-series patterns. KNN is generally unsuitable for time-series data, though DTW may improve back-testing results (but required much larger computational time)

<p float="left">
  <img src="/images/KNN_1.png" width="400" />
  <img src="/images/KNN_2.png" width="400" /> 
</p>

---

<h4>Support Vector Classification</h4>

Similar to Support Vector Regression, data was standardized, features were transformed using `StandardScaler`, and skewness was removed using power transformed. The best hyperparameters are {C: 1, gamma: 0.01, kernel: 'linear'}. SVC has accuracy of 0.857.

![SVM_class](/images/SVM_class.JPG)

---

<h4>Random Forest</h4>

For the `Base model`, I trained my data on `F`'s indicators alone, while for other models, I used both Ford’s and indicators from one company from the list of 6. I applied GridSearch for number of estimators, maximum depth, bootstrap, and criterion. `All` model was the model we used all indicators to train the data. Overall, `Base model`'s training accuracy of 0.546, which was slightly better than random guessing and other RF models using individual stock's information, and slightly worse than `All` model. parameters for both `All` and `Base` models were the same, but the `All` model had more features, which would improve the training accuracy.

| Ticker              | Number of estimators   | Maximum Depth | Bootstrap    | Criterion | Training Accuracy | Improvement (%) |
|---------------------|------------------------|---------------|-----------   |-----------|-------------------|-----------------|
| CAD/USD             | 100                    | 5             | False        | entropy   | 0.539             | -0.904          |
| GM                  | 500                    | 5             | True         | gini      | 0.544             | 0               |
| JCI                 | 500                    | 5             | False        | entropy   | 0.528             | -3.01           |
| TM                  | 100                    | 5             | True         | gini      | 0.531             | -2.41           |
| TRY/ USD            | 500                    | 5             | False        | entropy   | 0.538             | -1.20           |
| IXIC                | 80                     | 5             | False        | entropy   | 0.536             | -1.51           |
| All                 | 80                     | 5             | False        | entropy   | 0.546             | 0.301           |
| Base                | 80                     | 5             | False        | entropy   | 0.544             | 0               |

All training models were applied to the backtesting sets and walk forward validation sets. On the backtesting set, `RF` using `JCI` performed better than both `All` and `Base` model while the currecy exchange rate of `TRY/USD` performed the worst. However, in the walk-forward validation set, all models performed much better, where RF using `Nasdaq` indicators outperformed other models.

<p float="left">
  <img src="/images/ROC_backtesting.jpg" width="400" />
  <img src="/images/ROC_forward.jpg" width="400" /> 
</p>

---

<h2 id="#Investopedia-Simulation">Investopedia Simulation</h2>

To further evaluate our models, we created two trading games on the Investopedia Simulator. Each participant started with $100,000 and traded between November 4 and November 29, guided by the predictions of either continuous or classification models. Among the classification models, SVC delivered the best performance, while the regression model outperformed other continuous models in terms of predictive accuracy and trading outcomes.

<p float="center">
  <img src="/images/Investopedia_continuous.JPG" width="1000" />
  <img src="/images/Investopedia_class.JPG" width="1000" /> 
</p>

---

<h2 id="Model-Performance-Comparison">Model Performance Comparison</h2>

Our dataset was splitted up into training set and backtesting set. After training machine learning models on the training set, we validated our models on backtesting set. Regression and XGBoost models (with and without PCA) perform best.

<p float="center">
  <img src="/images/Backtesting_accuracy.jpg" width="1000" />
</p>

Despite the inherent market volatility, our models demonstrated reliable classification accuracy, providing valuable directional insights into stock price movements. 

To further validate their effectiveness, we tested the models using trading data from the specified time range. Among the approaches, XGBoost with PCA emerged as the top performer, followed by SVC and regression models.


| **Models**               | **Accuracy**           |      
|--------------------------|------------------------|
| XGBoost PCA              | 0.866                  |
| SVC                      | 0.857                  | 
| SVR                      | 0.84                   |
| XGBoost                  | 0.829                  | 
| Regression               | 0.829                  |
| Regression PCA           | 0.829                  |
| Random Forest            | 0.80                   |
| VAR                      | 0.789                  |
| LSTM                     | 0.789                  |
| KNN                      | 0.789                  |                     
---


<h2 id="Summary">Summary</h2>

- Efficient market theory suggests that historical data offers limited predictive power, often leading a model to behave like a random generator. However, by employing careful data manipulation, some of our models achieved strong market prediction power surpassing this limitation.
- We also observed that continuous and classification models have their own pros and cons.
    - Continuous models could predict continuous values, but it is sensitive to the market noise.
    - Classification models can predict trends more accurately, but it is less informative since it does not tell us about the exact price values.

---

<h2 id="Things-to-answer-and-to-be-updated-next">Things to answer and to be updated next</h2>

Our journey doesn’t stop here. We aim to make more accurate and actionable predictions by extending the analyses beyond stocks—to currencies, commodities, and beyond. Our project is the first step to explore and detect many more interesting factors that affect the volatility of the market data. One interesting factor is the influence of potential government assistance to companies, which we can explore through congressional trading data.

<h2 id="Code-description">Code Description</h2>

Data set can be found in the [Data](https://github.com/kpnguyen21/equity-vs-commodity/tree/main/Data) folder.
Notebooks containing various models used for results above can be found in the [Model](https://github.com/kpnguyen21/equity-vs-commodity/tree/main/Models) folder. The models are:
- [KNN.ipynb](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Models/KNN.ipynb)
- [SVM_reg](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Grishma's_NoteBook/Grishma_SVM_reg.ipynb)
- [SVM_class](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Grishma's_NoteBook/Grishma_SVM%20_class.ipynb)
- [VAR](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Models/VAR.ipynb)
- [RF](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Models/RF.ipynb)
- [LSTM](https://github.com/kpnguyen21/equity-vs-commodity/blob/main/Models/GC%20LSTM%20Cleaned.ipynb)
