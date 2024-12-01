# stock-vs-commodity
Erdos Data Science Bootcamp - Fall 2024

# Stock-Market-Prediction
A foundational step in predicting the stock market of a company by leveraging stock market prices of other companies and related commodities.
### Erdos Project

## Table of Contents

<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Project-details">Project Details</a></li>
    <li><a href="#Pre-analysis">Pre-analysis</a>
        <ul>
            <li><a href="#Time-Series">Time Series</a></li>
            <li><a href="#Financial-Factors">Financial Factors</a></li>
        </ul>
    </li>
    <li><a href="#Data">Data</a>
        <ul>
            <li><a href="#Data-Cleaning-and-Preparation">Data Cleaning and Preparation</a></li>
        </ul>
    </li>
    <li><a href="#Computer-experiments-to-study-patterns">Computer Experiments to Study Patterns</a></li>
    <li><a href="#Project-instructions">Project Instructions</a>
        <ul>
            <li><a href="#Steps">Steps</a>
                <ul>
                    <li><a href="#EDA-and-Preprocessing">EDA and Preprocessing</a></li>
                    <li><a href="#Models">Models</a></li>
                    <li><a href="#Prediction">Prediction</a></li>
                    <li><a href="#Performance">Performance</a></li>
                </ul>
            </li>
        </ul>
    </li>
    <li><a href="#Summary">Summary</a></li>
    <li><a href="#Things-to-answer-and-to-be-updated-next">Things to Answer and Update Next</a></li>
    <li><a href="#References">References</a></li>
</ul>

---

<h2 id="Data">Data</h2>
Our focus is on Ford (F), a major player in the U.S. automotive sector, as the testbed for this analysis. We consider 26 other companies that are directly or indirectly connected to Ford, including but not limited to companies in the same field and its supplier companies.

- **Automotive companies:** 'GM' (General Motors Co.), 'TM' (Toyota Motor Corp)
- **Supplier companies:** 'ALV' (Autoliv Inc), 'DNZOY' (Denso ADR), 'ASEKY' (Aisin Corp), 'JCI' (Johnson Controls International PLC)
- **Commodities:** 'CL=F' (Crude Oil), 'X' (US Steel Corporation), 'ALI=F' (Aluminum)
- **Currencies exchange rates:** 'EURUSD=X' (EUR/ USD), 'CHFUSD=X' (CHF/ USD), 'GBPUSD=X' (GBP/ USD), 'JPYUSD=X' (JPY/ USD), 'CADUSD=X' (CAD/ USD), 'INRUSD=X' (INR/ USD), 'RUBUSD=X' (RUB/ USD). 'TRYUSD=X' (TRY/ USD)
- **Financial companies and IRX:** '^IRX' (Inhalerx Ltd), 'JPM' (JPMorgan Chase & Co), 'BAC' (Bank of America Corp), 'C' (Citigroup Inc), 'WFC' (Wells Fargo & Co)
- **Others:** '^IXIC' (Nasdaq Composite), '^GSPC' (S&P), '^DJI' (Dow Jones Industrial Average), 'FDX' (FedEx Corp)

We found that 21 of 26 companies have strong correlations to Ford. Applying the Granger-causality test, we reduced our relevant companies to 6: CADUSD=X, GM, JCI, TM, TRYUSD=X, ^IXIC.

<h2 id="Summary">Summary</h2>

Efficient market theory suggests that historical data offers limited predictive power, often leading a model to behave like a random generator. However, by employing careful data manipulation, some of our models achieved strong market prediction power surpassing this limitation.

We also observed that continuous and classification models have their own pros and cons.
- Continuous models could predict continuous values, but it is sensitive to the market noise.
- Classification models can predict trends more accurately, but it is less informative since it does not tell us about the exact price values.


<h2 id="Things-to-answer-and-to-be-updated-next">Things to answer and to be updated next</h2>

Our journey doesn’t stop here. We aim to make more accurate and actionable predictions by extending the analyses beyond stocks—to currencies, commodities, and beyond. Our project is the first step to explore and detect many more interesting factors that affect the volatility of the market data. One interesting factor is the influence of potential government assistance to companies, which we can explore through congressional trading data.

