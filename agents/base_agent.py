import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple


class BaseAgent(ABC):
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0, 
                 risk_tolerance: float = 0.5):
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
        pass
    
    @abstractmethod
    def make_decision(self, market_data: pd.DataFrame, 
                      additional_info: Dict = None) -> Dict[str, Any]:
        pass
    
    def execute_trade(self, symbol: str, quantity: float, price: float, 
                      order_type: str = 'market') -> Dict[str, Any]:
        trade_value = quantity * price
        commission = abs(trade_value) * 0.001
        
        if quantity > 0:
            if self.cash >= trade_value + commission:
                self.cash -= (trade_value + commission)
                self.portfolio[symbol] = self.portfolio.get(symbol, 0) + quantity
                trade_status = 'executed'
            else:
                trade_status = 'rejected_insufficient_funds'
                return {'status': trade_status}
        else:
            if symbol in self.portfolio and self.portfolio[symbol] >= abs(quantity):
                self.cash += (abs(trade_value) - commission)
                self.portfolio[symbol] -= abs(quantity)
                if self.portfolio[symbol] == 0:
                    del self.portfolio[symbol]
                trade_status = 'executed'
            else:
                trade_status = 'rejected_insufficient_holdings'
                return {'status': trade_status}
        
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
        portfolio_value = self.cash
        for symbol, quantity in self.portfolio.items():
            if symbol in current_prices:
                portfolio_value += quantity * current_prices[symbol]
        return portfolio_value
    
    def calculate_returns(self, current_prices: Dict[str, float], 
                          initial_value: float) -> Dict[str, float]:
        current_value = self.get_portfolio_value(current_prices)
        total_return = (current_value - initial_value) / initial_value
        
        self.performance_metrics['total_return'] = total_return
        
        return self.performance_metrics
    
    def reset(self, initial_cash: float = 100000.0):
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