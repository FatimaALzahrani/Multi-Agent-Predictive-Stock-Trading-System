import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class FundamentalAgent(BaseAgent):
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.metrics = ['pe_ratio', 'pb_ratio', 'dividend_yield', 'earnings_growth', 'debt_to_equity']
        
        self.weight_pe = 0.25
        self.weight_pb = 0.20
        self.weight_dividend = 0.15
        self.weight_growth = 0.30
        self.weight_debt = 0.10
        
        self.pe_threshold = 20.0
        self.pb_threshold = 3.0
        self.dividend_threshold = 0.02
        self.growth_threshold = 0.10
        self.debt_threshold = 1.0
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].iloc[-1]
            
            try:
                pe_ratio = symbol_data.get('pe_ratio', 0)
                pb_ratio = symbol_data.get('pb_ratio', 0)
                dividend_yield = symbol_data.get('dividend_yield', 0)
                earnings_growth = symbol_data.get('earnings_growth', 0)
                debt_to_equity = symbol_data.get('debt_to_equity', 0)
                current_price = symbol_data.get('close', 0)
                
                pe_score = 1.0 - min(1.0, pe_ratio / self.pe_threshold) if pe_ratio > 0 else 0
                pb_score = 1.0 - min(1.0, pb_ratio / self.pb_threshold) if pb_ratio > 0 else 0
                dividend_score = min(1.0, dividend_yield / self.dividend_threshold)
                growth_score = min(1.0, earnings_growth / self.growth_threshold)
                debt_score = 1.0 - min(1.0, debt_to_equity / self.debt_threshold) if debt_to_equity > 0 else 1.0
                
                weighted_score = (
                    pe_score * self.weight_pe +
                    pb_score * self.weight_pb +
                    dividend_score * self.weight_dividend +
                    growth_score * self.weight_growth +
                    debt_score * self.weight_debt
                )
                
                results[symbol] = {
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'dividend_yield': dividend_yield,
                    'earnings_growth': earnings_growth,
                    'debt_to_equity': debt_to_equity,
                    'current_price': current_price,
                    'pe_score': pe_score,
                    'pb_score': pb_score,
                    'dividend_score': dividend_score,
                    'growth_score': growth_score,
                    'debt_score': debt_score,
                    'weighted_score': weighted_score
                }
                
            except Exception as e:
                results[symbol] = {
                    'error': str(e),
                    'weighted_score': 0.0
                }
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        buy_threshold = 0.6 + (0.2 * (1 - self.risk_tolerance))
        sell_threshold = 0.4 - (0.2 * (1 - self.risk_tolerance))
        
        for symbol, data in analysis.items():
            if 'error' in data:
                continue
                
            weighted_score = data['weighted_score']
            current_price = data['current_price']
            
            if weighted_score > buy_threshold:
                confidence = (weighted_score - buy_threshold) / (1.0 - buy_threshold)
                available_cash = self.cash * self.risk_tolerance * confidence
                quantity = int(available_cash / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    decisions[symbol] = {
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'confidence': confidence
                    }
            elif weighted_score < sell_threshold:
                if symbol in self.portfolio:
                    confidence = (sell_threshold - weighted_score) / sell_threshold
                    quantity = int(self.portfolio[symbol] * confidence)
                    
                    if quantity > 0:
                        decisions[symbol] = {
                            'action': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'confidence': confidence
                        }
            else:
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': current_price,
                    'confidence': 0.5
                }
        
        return decisions