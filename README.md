# Multi-Agent Predictive Stock Trading System

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A comprehensive multi-agent system for stock market trading and analysis, featuring four distinct intelligent agents with different trading strategies: Technical Analysis, Fundamental Analysis, Sentiment Analysis, and Reinforcement Learning.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Agent Types](#agent-types)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🔭 Overview

The Multi-Agent Predictive Stock Trading System is an educational and research platform that demonstrates the principles of multi-agent systems in the context of financial markets. The system simulates a stock trading environment where multiple intelligent agents with different strategies interact, make decisions, and learn from their experiences.

This project serves as both a learning tool for understanding multi-agent dynamics and a prototype for algorithmic trading research.

## 🏗️ System Architecture

The system follows a modular architecture with four main components:

```
+-------------------+       +-------------------+
|                   |       |                   |
|  Trading Agents   | <---> |  Market           |
|                   |       |  Environment      |
+-------------------+       +-------------------+
         ^                           ^
         |                           |
         v                           v
+-------------------+       +-------------------+
|                   |       |                   |
|  Dashboard        | <---> |  Data Analysis    |
|  Visualization    |       |  & Reporting      |
|                   |       |                   |
+-------------------+       +-------------------+
```

## ✨ Features

- **Multiple Trading Agents**: Four distinct agent types with different strategies
- **Market Simulation**: Realistic market environment with historical or synthetic data
- **Interactive Dashboard**: Real-time visualization of market data and agent performance
- **Performance Analytics**: Comprehensive metrics and comparison tools
- **Modular Design**: Easily extensible architecture for adding new agents or features
- **Synthetic Data Generation**: Built-in capability to generate realistic market data

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/FatimaALzahrani/Multi-Agent-Predictive-Stock-Trading-System
   cd multi-agent-trading-system
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

<!-- ### Running with Synthetic Data

For educational purposes or when external data sources are unavailable:

```bash
python src/main_dummy_complete.py
``` -->

<!-- ### Running with Real Market Data

For real market data (requires internet connection):

```bash
python src/main.py
``` -->

<!-- ### Accessing the Dashboard

Once the application is running, open your web browser and navigate to:
```
http://127.0.0.1:8050/
``` -->

### Dashboard Controls

- **Start**: Begin the simulation
- **Pause**: Temporarily halt the simulation
- **Reset**: Reset the simulation to the initial state
- **Speed**: Adjust the simulation speed using the slider

## 📁 Project Structure

```
Multi-Agent-Predictive-Stock-Trading-System/
├── agents/               # Trading agent implementations
│   ├── __init__.py
│   ├── base_agent.py     # Base agent class
│   ├── technical_agent.py
│   ├── fundamental_agent.py
│   ├── sentiment_agent.py
│   └── rl_agent.py
├── environment/          # Market environment
│   ├── __init__.py
│   ├── market.py         # Real market implementation
│   └── mock_market.py    # Synthetic market implementation
├── visualization/        # Dashboard and visualization
│   ├── __init__.py
│   └── dashboard.py
├── main.py               # Main entry point
├── data/                     # Data directory
│   ├── historical/           # Historical market data
│   ├── processed/            # Processed data
│   └── results/              # Simulation results
├── requirements.txt          # Project dependencies
├── README.md                 # This file
```

## 🤖 Agent Types

### Technical Analysis Agent

The Technical Analysis Agent uses price patterns and technical indicators to make trading decisions:

- Moving Averages (SMA, EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands

```python
# Example of Technical Analysis Agent decision logic
def make_decision(self, market_data):
    analysis = self.analyze_market(market_data)
    decisions = {}
    
    for symbol, data in analysis.items():
        technical_signal = data.get('signal', 'neutral')
        
        if technical_signal in ('buy', 'strong_buy'):
            # Calculate position size based on signal strength
            confidence = 0.7 if technical_signal == 'buy' else 0.9
            position_size = self.risk_tolerance * confidence
            # Execute buy decision
            # ...
```

### Fundamental Analysis Agent

The Fundamental Analysis Agent evaluates companies based on financial metrics:

- Price-to-Earnings (P/E) ratio
- Price-to-Book (P/B) ratio
- Return on Equity (ROE)
- Earnings growth
- Debt-to-Equity ratio

### Sentiment Analysis Agent

The Sentiment Analysis Agent uses market sentiment data:

- News sentiment
- Social media sentiment
- Analyst ratings
- Sentiment trends

### Reinforcement Learning Agent

The Reinforcement Learning Agent uses Q-learning to adapt its strategy:

- State representation of market conditions
- Action space for trading decisions
- Reward function based on profit/loss
- Exploration vs. exploitation balance

## 📊 Dashboard

The interactive dashboard provides:

- Real-time stock price charts
- Agent performance comparison
- Trading activity visualization
- Portfolio composition analysis
- Detailed agent information

## ⚙️ Configuration

The system can be configured through several parameters:

### Market Configuration

```python
# Example market configuration
market = MockMarketEnvironment(
    symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    start_date='2022-01-01',
    end_date='2022-12-31',
    data_source='synthetic' 
)
```

### Agent Configuration

```python
# Example agent configuration
technical_agent = TechnicalAgent(
    agent_id="technical_agent_1",
    name="Technical Analyst",
    initial_cash=100000.0,
    risk_tolerance=0.7
)
```

### Dashboard Configuration

```python
# Example dashboard configuration
dashboard = TradingDashboard(
    market=market,
    port=8050,
    theme='dark' 
)
```
