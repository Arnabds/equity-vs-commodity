# MarketMind : Unveiling Stock Trends with Machine Learning
Erdos Data Science Bootcamp - Fall 2024

A foundational step in predicting the stock market of a company by leveraging stock market prices of other companies and related commodities.

<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Project-details">Project Details</a></li>
    <li><a href="#Pre-analysis">Pre-analysis</a>
        <ul>
            <li><a href="#Time-Series">Time Series</a></li>    
        </ul>
    </li>
    <li><a href="#Data-Cleaning-and-Preparation">Data Cleaning and Preparation</a>
        <ul>
            <li><a href="#Data-Cleaning-and-Preparation">Data Cleaning and Preparation</a></li>
            <li><a href="#Financial-Factors">Financial Factors</a></li>
        </ul>
    </li>
    <li><a href="#Models">Models</a>
        <ul>
            <li><a href="#Continuous-Models">Continuous Models</a></li>
            <li><a href="#Classification-Models">Classification Models</a></li>
        </ul>
    </li>
    <li><a href="#Project-instructions">Project Instructions</a>
        <ul>
            <li><a href="#Steps">Steps</a>
                <ul>
                    <li><a href="#EDA-and-Preprocessing">EDA and Preprocessing</a></li>
                    <li><a href="#Models1">Models1</a></li>
                    <li><a href="#Prediction">Prediction</a></li>
                    <li><a href="#Performance">Performance</a></li>
                </ul>
            </li>
        </ul>
    </li>
    <li><a href="#Summary">Summary</a></li>
    <li><a href="#Things-to-answer-and-to-be-updated-next">Things to Answer and Update Next</a></li>
    <li><a href="#Code-description">Code Description</a></li>
</ul>

---
<h3 id="Pre-analysis">Pre-analysis</h3>

We began by studying the time series of Amazon stock prices and gold commodity prices. Using data imported from Yahoo Finance, we performed time series analyses. The available data included `Volume`, `High`, `Low`, `Open`, and `Close` prices, but we focused solely on `Close` prices for analysis.

We fitted regression models to estimate trends and removed these trends to analyze residuals for stationarity. Residual analyses involved the Ljung-Box test and the ADF Fuller test, revealing non-stationarity. Differencing was applied to achieve stationarity, paving the way for ARIMA modeling. The best-fitting models were ARIMA(1,1,2) for gold and ARIMA(1,2,0) for Amazon. Although we did not explore GARCH models for volatility prediction at this stage, this is a next step. This preliminary analysis helped us understand the challenges of stock market prediction.


<h2 id="Data">Data</h2>

Our focus is on Ford (`'F'`), a major player in the U.S. automotive sector, as the testbed for this analysis. We consider 26 other companies that are directly or indirectly connected to Ford, including but not limited to companies in the same field and its supplier companies.

- **Automotive companies:** `'GM'` (General Motors Co.), `'TM'` (Toyota Motor Corp)
- **Supplier companies:** `'ALV'` (Autoliv Inc), `'DNZOY'` (Denso ADR), `'ASEKY'` (Aisin Corp), `'JCI'` (Johnson Controls International PLC)
- **Commodities:** `'CL=F'` (Crude Oil), `'X'` (US Steel Corporation), `'ALI=F'` (Aluminum)
- **Currencies exchange rates:** `'EURUSD=X'` (EUR/ USD), `'CHFUSD=X'` (CHF/ USD), `'GBPUSD=X'` (GBP/ USD), `'JPYUSD=X'` (JPY/ USD), `'CADUSD=X'` (CAD/ USD), `'INRUSD=X'` (INR/ USD), `'RUBUSD=X'` (RUB/ USD). `'TRYUSD=X'` (TRY/ USD)
- **Financial companies and IRX:** `'^IRX'` (Inhalerx Ltd), `'JPM'` (JPMorgan Chase & Co), `'BAC'` (Bank of America Corp),`'C'` (Citigroup Inc), `'WFC'` (Wells Fargo & Co)
- **Others:** `'^IXIC'` (Nasdaq Composite), `'^GSPC'` (S&P), `'^DJI'` (Dow Jones Industrial Average), `'FDX'` (FedEx Corp)

<h3 id="Data-Cleaning-and-Preparation">Data Cleaning and Preparation</h3>

We found that 21 of 26 companies have strong correlations to Ford. Applying the Granger-causality test, we reduced our relevant companies to 6. The final set of features included: `'CADUSD=X'`, `'GM'`, `'JCI'`, `'TM'`, `'TRYUSD=X'`, `'^IXIC'`, and `'F'`. 

<img src=/images/data.png width="800" class="center" />

We assessed model performance for continuous and categorical prediction using continuous model such as LSTM, and Classification model like Logistic Regression. We employed forward cross-validation and backtesting to evaluate model robustness.
<img src=/images/preprocessing.jpg width="800" class="center" />

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
- **Exponential Moving Average** calculates exponential average of close prices over a period 20 days and weighted average that gives greater importance to the price in more recent days. 
- **Rate of Change** measures the most recent change in price with respect to the price 12 days ago.
- **Price Volume Trend** determines a security's price direction and strength of price change.


---
<h2 id="Models">Models</h2>

- This project has 3 primary goals:
  - Train a model that accurately predicts the closing price value.
  - Train a model that accurately classify whether the stock closing price will go up or down the next day.
  - Invest fake money in investopedia to check our model's prediction performance.
- We must use an appropriate validation scheme to select the best model!

<h3 id="Continuous-Models">Continuous Models</h3>

<h4>Regression and XGBoost Models</h4>
We have fitted 4 models from linear regression, linear regression with PCA, linear logistic regression, linear logistic regression with PCA all with regularization, XGBoost, XGBoost PCA.

![XGBoost regression](/images/XGboost_reg.png)

![XGBoost regression pca](/images/XGBoost_reg_pca.png)

![Regression](/images/Reg.png)

<h4>LSTM</h4>

A neural network model can be effectively employed for this purpose, with Recurrent Neural Networks being particularly well-suited for time series modeling.
Among RNNs, Long Short-Term Memory modules are especially advantageous for capturing temporal dependencies.
Unlike many other classical models, this approach eliminates the need for preselecting input features, simplifying the process.
During the training phase, historical data from highly correlated tickers organized based on Granger causality, and their various indicator functions, is used as input.
The input data is processed through a stacked LSTM module and a fully connected layer in turn.
Finally, the model produces a continuous output value representing the predicted stock price of interest.

![LSTM](/images/LSTM.JPG)

<h3 id="Classification-Models">Classification Models</h3>




<h2 id="Steps">Steps</h2>
We have divided our project into 6 parts: ***EDA and Preprocessing, Cluster Analysis, Models, Performance, Prediction, and Testing***. We divided our work in groups. You will get these files with codes in jupyter notebook and HTML folders in Github. Introduction has been given so far. Let us start with the EDA.

|EDA and Preprocessing|Cluster Analysis|Models|Performance|Prediction|Bonus|
|--------|--------|--------|--------|--------|--------|
|Plotting necessary data, Standardization, Removing skewness, PCA|KMeans, Hierarchical clustering|7 logistic regression models and accuracy on training data|testing on manually created data|Gridsearch, lasso, ridge, elastic net|SVC, Neural net|
|Python|Python|Python|Python|Python|Python|



<h2 id="EDA-and-Preprocessing">EDA and Preprocessing</h2>

![Overall model with bollinger band](/images/Overall_model_with_bollinger_band.png)


This is the correlation plot with all the final selected features.
![Correlation plot](/images/corr.png)


<h2 id="Models">Models</h2>
We have fitted 4 models from linear regression, linear regression with PCA, linear logistic regression, linear logistic regression with PCA all with regularization, XGBoost, XGBoost PCA.


<h2 id="Summary">Summary</h2>

- Efficient market theory suggests that historical data offers limited predictive power, often leading a model to behave like a random generator. However, by employing careful data manipulation, some of our models achieved strong market prediction power surpassing this limitation.
- We also observed that continuous and classification models have their own pros and cons.
    - Continuous models could predict continuous values, but it is sensitive to the market noise.
    - Classification models can predict trends more accurately, but it is less informative since it does not tell us about the exact price values.


<h2 id="Things-to-answer-and-to-be-updated-next">Things to answer and to be updated next</h2>

Our journey doesn’t stop here. We aim to make more accurate and actionable predictions by extending the analyses beyond stocks—to currencies, commodities, and beyond. Our project is the first step to explore and detect many more interesting factors that affect the volatility of the market data. One interesting factor is the influence of potential government assistance to companies, which we can explore through congressional trading data.

<h2 id="Code-description">Code Description</h2>

Notebooks containing various models used for results above can be found in this [folder](https://github.com/kpnguyen21/equity-vs-commodity/tree/main/Models).
