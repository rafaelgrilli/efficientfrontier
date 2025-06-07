# efficientfrontier
📈 Efficient Frontier (EF) Simulation – Stanford's Code in Place 2025!  I combined new Python skills with my data science &amp; finance background to build an interactive tool. It runs Monte Carlo simulations to visualize the EF and the Minimum Variance Portfolio. Beyond the course, I integrated advanced libraries for financial data &amp; interactivity.


# 📈 Efficient Frontier Simulator – Code in Place Final Project (Stanford 2025)

This project is the final deliverable of my journey through Stanford's [Code in Place](https://codeinplace.stanford.edu/) 2025 course. It merges my newly acquired Python programming skills with my professional background in finance to simulate investment portfolios and visualize the **Efficient Frontier** using real market data.

## 🧠 Project Summary

The simulator performs a **Monte Carlo simulation** of thousands of random portfolios based on selected assets. It calculates the expected return and risk (volatility) of each portfolio, identifies the **Minimum Variance Portfolio**, and displays the results graphically in an **Efficient Frontier** chart.

## 💼 Background Integration

As a finance professional, I work regularly with concepts such as portfolio optimization and risk-return analysis. This project is where theory meets code. It brings financial models to life using Python, leveraging data science tools to generate insights and simulate strategies.

## 🚀 Features

- Fetch historical adjusted prices via Yahoo Finance
- Clean and prepare time series data
- Calculate expected returns, volatility, and correlations
- Generate thousands of random portfolios
- Plot the Efficient Frontier using `matplotlib`
- Highlight the Minimum Variance Portfolio
- Interactive user interface with `ipywidgets`
- All computations in real-time within a Jupyter Notebook

## 🧰 Technologies and Libraries

This project goes **beyond the Code in Place curriculum** by integrating several external libraries to handle real-world data, visualizations, and interactivity:

| Library         | Purpose |
|----------------|---------|
| [`yahooquery`](https://pypi.org/project/yahooquery/) | Fetch live financial data from Yahoo Finance |
| `pandas`        | Data manipulation and time series handling |
| `numpy`         | Numerical operations and matrix algebra |
| `matplotlib`    | Plotting the Efficient Frontier |
| `ipywidgets`    | Interactive user input (tickers, dates, etc.) |
| `IPython.display` | Clear and update output inside the notebook |

## 📸 Example of chart output

![Efficient Frontier](https://github.com/user-attachments/assets/9ddf3246-5945-419c-a831-9340adde7a07)


## 📋 Usage

1. Clone the repository or open the notebook in Jupyter:
   ```bash
   git clone https://github.com/your-username/efficient-frontier-simulator.git
   cd efficient-frontier-simulator
