#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 📦 Import necessary libraries
from yahooquery import Ticker               # To fetch stock data from Yahoo Finance
import pandas as pd                         # For data manipulation
import numpy as np                          # For numerical operations
import matplotlib.pyplot as plt             # For plotting the efficient frontier
import ipywidgets as widgets                # For interactive widgets in Jupyter
from IPython.display import display, clear_output  # For displaying and clearing outputs in notebook

# 🎯 Supporting Functions

def random_weights(n):
    """
    Generate random weights for n assets that sum to 1.
    Used to simulate random portfolio allocations.
    """
    weights = np.random.rand(n)             # Generate n random values between 0 and 1
    return weights / sum(weights)           # Normalize so the total is 1 (100% allocation)

def portfolio_return(weights, expected_returns):
    """
    Calculate the expected annual return of a portfolio.
    """
    return np.dot(weights, expected_returns)  # Weighted sum of expected returns

def portfolio_risk(weights, std_devs, correlation_matrix):
    """
    Calculate the portfolio risk (standard deviation of returns).
    Uses the formula: sqrt(W.T * CovMatrix * W)
    """
    # Construct covariance matrix from std deviations and correlations
    cov_matrix = np.outer(std_devs, std_devs) * correlation_matrix
    # Compute portfolio variance
    variance = np.dot(weights, np.dot(cov_matrix, weights))
    return np.sqrt(variance)  # Return standard deviation (risk)

# 🧠 Main Functionality Wrapped in One Interactive Function

def run_simulation(tickers_input, start_date, end_date, num_portfolios):
    """
    Main function to perform portfolio simulation using user inputs.
    """
    clear_output(wait=True)  # Clear previous output for a cleaner interface
    print("📊 Fetching data and running simulation...\n")

    # 🎯 Process ticker input: split by commas and convert to uppercase
    tickers = [t.strip().upper() for t in tickers_input.split(",")]

    # 🔍 Fetch historical price data using yahooquery
    ticker_obj = Ticker(" ".join(tickers))
    df = ticker_obj.history(start=start_date, end=end_date)

    # Check if data is returned
    if df.empty:
        print("❌ No data found. Check tickers or date range.")
        return

    # 🧹 Prepare price data
    # Reset multi-index and reshape so each column is a ticker
    prices = df.reset_index().pivot(index='date', columns='symbol', values='adjclose')
    prices = prices.ffill().dropna()  # Forward-fill missing values and drop remaining NaNs

    # 📈 Calculate daily returns and drop missing rows
    returns = prices.pct_change().dropna()

    # 📊 Calculate annualized metrics
    expected_returns = returns.mean() * 252            # Expected return = mean daily return × 252
    std_devs = returns.std() * np.sqrt(252)            # Annualized standard deviation
    correlation_matrix = returns.corr().values         # Correlation matrix between assets

    # 🔍 Display basic statistics
    print("📈 Expected Annual Returns (%):")
    print((expected_returns * 100).round(2))           # Display as percentages
    print("\n📊 Annual Volatility (%):")
    print((std_devs * 100).round(2))                   # Display as percentages
    print("\n🔗 Correlation Matrix:")
    print(pd.DataFrame(correlation_matrix, index=tickers, columns=tickers))

    # 🎲 Monte Carlo Simulation to generate random portfolios
    num_assets = len(tickers)
    portfolio_returns = []
    portfolio_risks = []
    portfolio_weights = []

    for _ in range(num_portfolios):
        weights = random_weights(num_assets)                          # Generate random weights
        ret = portfolio_return(weights, expected_returns)             # Calculate return
        risk = portfolio_risk(weights, std_devs, correlation_matrix)  # Calculate risk

        portfolio_returns.append(ret)
        portfolio_risks.append(risk)
        portfolio_weights.append(weights)

    # 🥇 Identify the minimum variance portfolio (lowest risk)
    min_index = portfolio_risks.index(min(portfolio_risks))
    min_risk = portfolio_risks[min_index]
    min_return = portfolio_returns[min_index]
    min_weights = portfolio_weights[min_index]

    # 🎨 Plot the Efficient Frontier
    plt.figure(figsize=(10, 6))
    plt.scatter(np.array(portfolio_risks) * 100, np.array(portfolio_returns) * 100,
                c='lightblue', label='Portfolios')
    plt.scatter(min_risk * 100, min_return * 100, c='red', marker='*', s=200,
                label='Minimum Variance')  # Highlight min variance portfolio
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (%)')
    plt.ylabel('Expected Return (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 📜 Display minimum variance portfolio allocation
    print("\n🌟 Minimum Variance Portfolio Allocation:")
    for name, weight in zip(tickers, min_weights):
        print(f"  {name}: {weight*100:.2f}%")         # Show percentage allocation per asset
    print(f"Expected Return: {min_return*100:.2f}%")
    print(f"Volatility (Risk): {min_risk*100:.2f}%")
    print("\n✅ Simulation completed.\n")

# 🧰 Widgets for User Interaction

# Text box for ticker symbols input
ticker_input = widgets.Text(
    value='ITSA4.SA, VALE3.SA, PETR4.SA',
    description='Tickers:',
    placeholder='Enter tickers separated by commas',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px')
)

# Text box for start date input
start_date = widgets.Text(
    value='2018-01-01',
    description='Start Date (YYYY-MM-DD):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# Text box for end date input
end_date = widgets.Text(
    value='2024-12-31',
    description='End Date (YYYY-MM-DD):',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='300px')
)

# Slider to select the number of portfolios to simulate
num_portfolios = widgets.IntSlider(
    value=5000,
    min=1000,
    max=20000,
    step=1000,
    description='Simulations:',
    continuous_update=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

# Run button to trigger the simulation
run_button = widgets.Button(
    description="🚀 Run Simulation",
    button_style='success',
    layout=widgets.Layout(width='200px')
)

# 🚦 Event handler to run the simulation when button is clicked
def on_run_button_clicked(b):
    run_simulation(ticker_input.value, start_date.value, end_date.value, num_portfolios.value)

# Link the button click to the handler
run_button.on_click(on_run_button_clicked)

# 🎯 Display the input widgets in the notebook
display(ticker_input, start_date, end_date, num_portfolios, run_button)


# In[ ]:




