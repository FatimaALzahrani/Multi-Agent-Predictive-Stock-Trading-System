"""
Mock Market Environment Module for Multi-Agent Trading System

This module implements a simulated market environment with dummy data where agents interact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import datetime


class MockMarketEnvironment:
    """
    Simulated market environment with dummy data for trading agents.
    
    Attributes:
        symbols (List[str]): List of stock symbols in the market
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        current_step (int): Current time step in the simulation
        data (pd.DataFrame): Generated dummy market data
        current_prices (Dict[str, float]): Current prices for all symbols
        agents (List): List of trading agents in the simulation
    """
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str = None):
        """
        Initialize the mock market environment.
        
        Args:
            symbols: List of stock symbols to include
            start_date: Start date for historical data (YYYY-MM-DD)
            end_date: End date for historical data (YYYY-MM-DD), defaults to today
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.now().strftime('%Y-%m-%d')
        self.current_step = 0
        self.data = None
        self.current_prices = {}
        self.agents = []
        self.daily_returns = {}
        self.market_history = []
        
        # Generate dummy data
        self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        """Generate dummy market data for all symbols."""
        all_data = []
        
        # Convert start and end dates to datetime
        start_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Calculate number of days
        days = (end_date - start_date).days + 1
        
        # Generate date range
        date_range = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        # Generate data for each symbol
        for symbol in self.symbols:
            # Set initial price based on symbol (just for variety)
            if symbol == 'AAPL':
                initial_price = 150.0
            elif symbol == 'MSFT':
                initial_price = 300.0
            elif symbol == 'GOOGL':
                initial_price = 2500.0
            elif symbol == 'AMZN':
                initial_price = 3000.0
            elif symbol == 'TSLA':
                initial_price = 700.0
            else:
                initial_price = 100.0
            
            # Generate price series with random walk
            prices = [initial_price]
            for i in range(1, days):
                # Random daily return between -3% and +3%
                daily_return = np.random.normal(0.0005, 0.015)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
            
            # Create DataFrame for this symbol
            symbol_data = pd.DataFrame({
                'date': date_range,
                'symbol': symbol,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                'close': prices,
                'adj close': prices,
                'volume': [int(np.random.uniform(1000000, 10000000)) for _ in range(days)]
            })
            
            all_data.append(symbol_data)
        
        # Combine all data
        self.data = pd.concat(all_data, ignore_index=True)
        
        # Calculate additional features
        self._calculate_features()
    
    def _calculate_features(self):
        """Calculate additional features for analysis."""
        # Group by symbol to calculate features for each stock
        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            if not symbol_data.empty:
                # Sort by date
                symbol_data = symbol_data.sort_values('date')
                
                # Calculate daily returns
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
                
                # Calculate technical indicators
                # SMA
                symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                
                # EMA
                symbol_data['ema_20'] = symbol_data['close'].ewm(span=20, adjust=False).mean()
                
                # RSI
                delta = symbol_data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                # MACD
                ema_12 = symbol_data['close'].ewm(span=12, adjust=False).mean()
                ema_26 = symbol_data['close'].ewm(span=26, adjust=False).mean()
                symbol_data['macd'] = ema_12 - ema_26
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean()
                
                # Bollinger Bands
                sma_20 = symbol_data['close'].rolling(window=20).mean()
                std_20 = symbol_data['close'].rolling(window=20).std()
                symbol_data['upper_band'] = sma_20 + (std_20 * 2)
                symbol_data['lower_band'] = sma_20 - (std_20 * 2)
                
                # Add sentiment score (dummy)
                symbol_data['sentiment_score'] = np.random.uniform(0.3, 0.7, len(symbol_data))
                
                # Update the main dataframe
                self.data.update(symbol_data)
    
    def add_agent(self, agent):
        """
        Add a trading agent to the simulation.
        
        Args:
            agent: Trading agent to add
        """
        self.agents.append(agent)
    
    def get_current_data(self) -> pd.DataFrame:
        """
        Get market data for the current time step.
        
        Returns:
            DataFrame with current market data
        """
        # Get unique dates in the data
        dates = sorted(self.data['date'].unique())
        
        if self.current_step < len(dates):
            current_date = dates[self.current_step]
            current_data = self.data[self.data['date'] == current_date].copy()
            
            # Update current prices
            for symbol in self.symbols:
                symbol_data = current_data[current_data['symbol'] == symbol]
                if not symbol_data.empty:
                    self.current_prices[symbol] = symbol_data['close'].iloc[0]
            
            return current_data
        else:
            # End of simulation
            return pd.DataFrame()
    
    def get_historical_window(self, window_size: int = 30) -> pd.DataFrame:
        """
        Get historical market data for a window before the current step.
        
        Args:
            window_size: Number of days to include in the window
            
        Returns:
            DataFrame with historical market data
        """
        # Get unique dates in the data
        dates = sorted(self.data['date'].unique())
        
        if self.current_step < len(dates):
            # Calculate start index (ensure it's not negative)
            start_idx = max(0, self.current_step - window_size + 1)
            
            # Get date range
            date_range = dates[start_idx:self.current_step + 1]
            
            # Filter data by date range
            historical_data = self.data[self.data['date'].isin(date_range)].copy()
            
            return historical_data
        else:
            # End of simulation
            return pd.DataFrame()
    
    def step(self) -> Dict[str, Any]:
        """
        Advance the simulation by one time step.
        
        Returns:
            Dictionary with simulation state information
        """
        # Get current market data
        current_data = self.get_current_data()
        
        if current_data.empty:
            # End of simulation
            return {'done': True}
        
        # Get historical data for analysis
        historical_data = self.get_historical_window()
        
        # Record market state
        market_state = {
            'date': current_data['date'].iloc[0],
            'prices': self.current_prices.copy(),
            'agent_actions': {}
        }
        
        # Let each agent analyze and make decisions
        for agent in self.agents:
            # Agent analyzes market
            analysis = agent.analyze_market(historical_data)
            
            # Agent makes decisions
            decisions = agent.make_decision(historical_data)
            
            # Execute trades
            agent_actions = {}
            for symbol, decision in decisions.items():
                if decision['action'] in ['buy', 'sell'] and decision['quantity'] > 0:
                    trade_result = agent.execute_trade(
                        symbol=symbol,
                        quantity=decision['quantity'] if decision['action'] == 'buy' else -decision['quantity'],
                        price=self.current_prices.get(symbol, 0)
                    )
                    agent_actions[symbol] = {
                        'action': decision['action'],
                        'quantity': decision['quantity'],
                        'price': self.current_prices.get(symbol, 0),
                        'result': trade_result
                    }
            
            # Record agent actions
            market_state['agent_actions'][agent.agent_id] = agent_actions
            
            # Calculate agent performance
            agent.calculate_returns(self.current_prices, 100000)  # Assuming initial value of 100,000
        
        # Record market state
        self.market_history.append(market_state)
        
        # Move to next time step
        self.current_step += 1
        
        return {
            'done': False,
            'current_step': self.current_step,
            'date': current_data['date'].iloc[0],
            'prices': self.current_prices.copy(),
            'agent_actions': market_state['agent_actions']
        }
    
    def run_simulation(self, steps: int = None) -> List[Dict[str, Any]]:
        """
        Run the simulation for a specified number of steps or until completion.
        
        Args:
            steps: Number of steps to run, or None to run until end of data
            
        Returns:
            List of simulation state dictionaries
        """
        results = []
        done = False
        step_count = 0
        
        while not done and (steps is None or step_count < steps):
            step_result = self.step()
            results.append(step_result)
            done = step_result['done']
            step_count += 1
        
        return results
    
    def get_agent_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all agents.
        
        Returns:
            Dictionary mapping agent IDs to performance metrics
        """
        performance = {}
        
        for agent in self.agents:
            performance[agent.agent_id] = agent.performance_metrics
        
        return performance
    
    def reset(self):
        """Reset the simulation to the beginning."""
        self.current_step = 0
        self.market_history = []
        
        # Reset all agents
        for agent in self.agents:
            agent.reset()
