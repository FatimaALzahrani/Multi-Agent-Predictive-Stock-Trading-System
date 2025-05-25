"""
Base Agent Module for Multi-Agent Trading System

This module defines the base agent class that all trading agents will inherit from.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents in the system.
    
    Attributes:
        agent_id (str): Unique identifier for the agent
        name (str): Human-readable name for the agent
        cash (float): Current cash holdings
        portfolio (Dict[str, float]): Current portfolio holdings {symbol: quantity}
        risk_tolerance (float): Risk tolerance level (0-1)
        transaction_history (List): History of all transactions
        performance_metrics (Dict): Dictionary of performance metrics
    """
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0, 
                 risk_tolerance: float = 0.5):
        """
        Initialize the base agent with starting parameters.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            initial_cash: Starting cash amount
            risk_tolerance: Risk tolerance level (0-1)
        """
        self.agent_id = agent_id
        self.name = name
        self.cash = initial_cash
        self.portfolio = {}
        self.risk_tolerance = risk_tolerance
        self.transaction_history = []
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'trades_count': 0
        }
    
    @abstractmethod
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data and return analysis results.
        
        Args:
            market_data: DataFrame containing market data
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def make_decision(self, market_data: pd.DataFrame, 
                      additional_info: Dict = None) -> Dict[str, Any]:
        """
        Make trading decision based on market data and analysis.
        
        Args:
            market_data: DataFrame containing market data
            additional_info: Additional information that might be useful
            
        Returns:
            Dictionary containing trading decisions
        """
        pass
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                      order_type: str = 'market') -> Dict[str, Any]:
        """
        Execute a trade and update portfolio and cash.
        
        Args:
            symbol: Stock symbol
            quantity: Quantity to trade (positive for buy, negative for sell)
            price: Execution price
            order_type: Type of order ('market', 'limit', etc.)
            
        Returns:
            Dictionary containing trade details
        """
        trade_value = quantity * price
        commission = abs(trade_value) * 0.001  # 0.1% commission
        
        if quantity > 0:  # Buy order
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                trade_status = 'executed'
            else:
                trade_status = 'rejected_insufficient_funds'
                return {'status': trade_status}
        else:  # Sell order
            if symbol in self.portfolio and self.portfolio[symbol] >= abs(quantity):
                self.cash += (abs(trade_value) - commission)
                self.portfolio[symbol] -= abs(quantity)
                if self.portfolio[symbol] == 0:
                    del self.portfolio[symbol]
                trade_status = 'executed'
            else:
                trade_status = 'rejected_insufficient_holdings'
                return {'status': trade_status}
        
        # Record the transaction
        transaction = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'commission': commission,
            'type': 'buy' if quantity > 0 else 'sell',
            'order_type': order_type,
            'status': trade_status
        }
        self.transaction_history.append(transaction)
        self.performance_metrics['trades_count'] += 1
        
        return transaction
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate the current portfolio value.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            
        Returns:
            Total portfolio value including cash
        """
        portfolio_value = self.cash
        for symbol, quantity in self.portfolio.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
    
    def calculate_returns(self, current_prices: Dict[str, float], 
                          initial_value: float) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            current_prices: Dictionary mapping symbols to current prices
            initial_value: Initial portfolio value
            
        Returns:
            Updated performance metrics
        """
        current_value = self.get_portfolio_value(current_prices)
        total_return = (current_value - initial_value) / initial_value
        
        # Update performance metrics
        self.performance_metrics['total_return'] = total_return
        
        # Additional metrics would be calculated here in a real implementation
        # Such as Sharpe ratio, max drawdown, etc.
        
        return self.performance_metrics
    
    def reset(self, initial_cash: float = 100000.0):
        """
        Reset the agent to initial state.
        
        Args:
            initial_cash: Starting cash amount
        """
        self.cash = initial_cash
        self.portfolio = {}
        self.transaction_history = []
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'trades_count': 0
        }
