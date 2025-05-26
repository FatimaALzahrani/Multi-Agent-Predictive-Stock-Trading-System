"""
Main entry point for Multi-Agent Predictive Stock Trading System with Dummy Data

This module integrates all components and runs the simulation with dashboard using dummy data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import agent modules
from agents.base_agent import BaseAgent
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
# Import the fixed sentiment agent instead of the original one
from agents.sentiment_agent import SentimentAgent
# Import the fixed RL agent instead of the original one
from agents.rl_agent import RLAgent

# Import environment module - using mock market instead of real market
from environment.market import MockMarketEnvironment

# Import visualization module
from visualization.dashboard import TradingDashboard


def main():
    """Main function to run the trading system with dummy data."""
    print("Starting Multi-Agent Predictive Stock Trading System with Dummy Data...")
    
    # Define stock symbols to include in the simulation
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Define date range for historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f"Initializing mock market environment with symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    
    # Create mock market environment with dummy data
    market = MockMarketEnvironment(symbols=symbols, start_date=start_date, end_date=end_date)
    
    # Create agents
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
    
    # Add agents to market
    market.add_agent(technical_agent)
    market.add_agent(fundamental_agent)
    market.add_agent(sentiment_agent)
    market.add_agent(rl_agent)
    
    # Create and run dashboard
    print("Starting dashboard...")
    dashboard = TradingDashboard(market=market, port=8050)
    dashboard.run(debug=True, host='0.0.0.0')



if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs("data/historical", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/results", exist_ok=True)
    
    # Run the main function
    main()
