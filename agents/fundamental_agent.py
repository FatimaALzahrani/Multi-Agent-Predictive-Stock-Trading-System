"""
Fundamental Analysis Agent Module for Multi-Agent Trading System

This module implements a trading agent that uses fundamental analysis for decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class FundamentalAgent(BaseAgent):
    """
    Trading agent that uses fundamental analysis to make decisions.
    
    Attributes:
        metrics (List[str]): List of fundamental metrics to consider
        weight_pe (float): Weight for P/E ratio in decision making
        weight_pb (float): Weight for P/B ratio in decision making
        weight_dividend (float): Weight for dividend yield in decision making
        weight_growth (float): Weight for earnings growth in decision making
    """
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        """
        Initialize the fundamental analysis agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            initial_cash: Starting cash amount
            risk_tolerance: Risk tolerance level (0-1)
        """
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.metrics = ['pe_ratio', 'pb_ratio', 'dividend_yield', 'earnings_growth', 'debt_to_equity']
        
        # Weights for different fundamental metrics
        self.weight_pe = 0.25
        self.weight_pb = 0.20
        self.weight_dividend = 0.15
        self.weight_growth = 0.30
        self.weight_debt = 0.10
        
        # Thresholds for metrics
        self.pe_threshold = 20.0  # P/E ratio threshold
        self.pb_threshold = 3.0   # P/B ratio threshold
        self.dividend_threshold = 0.02  # Dividend yield threshold (2%)
        self.growth_threshold = 0.10  # Earnings growth threshold (10%)
        self.debt_threshold = 1.0  # Debt-to-equity threshold
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using fundamental metrics.
        
        Args:
            market_data: DataFrame with fundamental data for stocks
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Group by symbol to analyze each stock
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].iloc[-1]  # Get latest data
            
            # Extract fundamental metrics
            try:
                pe_ratio = symbol_data.get('pe_ratio', 0)
                pb_ratio = symbol_data.get('pb_ratio', 0)
                dividend_yield = symbol_data.get('dividend_yield', 0)
                earnings_growth = symbol_data.get('earnings_growth', 0)
                debt_to_equity = symbol_data.get('debt_to_equity', 0)
                current_price = symbol_data.get('close', 0)
                
                # Calculate scores for each metric (lower is better for PE, PB, and debt)
                pe_score = 1.0 - min(1.0, pe_ratio / self.pe_threshold) if pe_ratio > 0 else 0
                pb_score = 1.0 - min(1.0, pb_ratio / self.pb_threshold) if pb_ratio > 0 else 0
                dividend_score = min(1.0, dividend_yield / self.dividend_threshold)
                growth_score = min(1.0, earnings_growth / self.growth_threshold)
                debt_score = 1.0 - min(1.0, debt_to_equity / self.debt_threshold) if debt_to_equity > 0 else 1.0
                
                # Calculate weighted score
                weighted_score = (
                    pe_score * self.weight_pe +
                    pb_score * self.weight_pb +
                    dividend_score * self.weight_dividend +
                    growth_score * self.weight_growth +
                    debt_score * self.weight_debt
                )
                
                # Store results
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
                # Handle missing data
                results[symbol] = {
                    'error': str(e),
                    'weighted_score': 0.0
                }
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        """
        Make trading decisions based on fundamental analysis.
        
        Args:
            market_data: DataFrame containing market data with fundamental metrics
            additional_info: Additional information that might be useful
            
        Returns:
            Dictionary containing trading decisions
        """
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        # Decision threshold based on risk tolerance
        buy_threshold = 0.6 + (0.2 * (1 - self.risk_tolerance))  # Higher threshold for lower risk tolerance
        sell_threshold = 0.4 - (0.2 * (1 - self.risk_tolerance))  # Lower threshold for lower risk tolerance
        
        for symbol, data in analysis.items():
            if 'error' in data:
                continue  # Skip stocks with missing data
                
            weighted_score = data['weighted_score']
            current_price = data['current_price']
            
            # Make decision based on weighted score
            if weighted_score > buy_threshold:
                # Calculate position size based on score and risk tolerance
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
                # Sell based on poor fundamental score
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
                # Hold position
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': current_price,
                    'confidence': 0.5  # Neutral confidence
                }
        
        return decisions
