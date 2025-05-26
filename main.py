import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from agents.base_agent import BaseAgent
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentiment_agent import SentimentAgent
from agents.rl_agent import RLAgent

from environment.market import MockMarketEnvironment
from visualization.dashboard import TradingDashboard


def main():
    print("Starting Multi-Agent Predictive Stock Trading System with Dummy Data...")
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Initializing mock market environment with symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    
    market = MockMarketEnvironment(symbols=symbols, start_date=start_date, end_date=end_date)
    
    print("Creating trading agents...")
    
    technical_agent = TechnicalAgent(
        agent_id="technical_agent_1",
        name="Technical Analyst",
        initial_cash=100000.0,
        risk_tolerance=0.7
    )
    
    fundamental_agent = FundamentalAgent(
        agent_id="fundamental_agent_1",
        name="Fundamental Analyst",
        initial_cash=100000.0,
        risk_tolerance=0.5
    )
    
    sentiment_agent = SentimentAgent(
        agent_id="sentiment_agent_1",
        name="Sentiment Analyst",
        initial_cash=100000.0,
        risk_tolerance=0.6
    )
    
    rl_agent = RLAgent(
        agent_id="rl_agent_1",
        name="AI Trader",
        initial_cash=100000.0,
        risk_tolerance=0.8
    )
    
    market.add_agent(technical_agent)
    market.add_agent(fundamental_agent)
    market.add_agent(sentiment_agent)
    market.add_agent(rl_agent)
    
    print("Starting dashboard...")
    dashboard = TradingDashboard(market=market, port=8050)
    dashboard.run(debug=True, host='0.0.0.0')



if __name__ == "__main__":
    os.makedirs("data/historical", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    main()