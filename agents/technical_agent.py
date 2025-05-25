"""
Technical Analysis Agent Module for Multi-Agent Trading System

This module implements a trading agent that uses technical indicators for decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class TechnicalAgent(BaseAgent):
    """
    Trading agent that uses technical analysis indicators to make decisions.
    
    Attributes:
        indicators (List[str]): List of technical indicators to use
        lookback_period (int): Historical data period to consider
        signal_threshold (float): Threshold for generating buy/sell signals
    """
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5, lookback_period: int = 20,
                 signal_threshold: float = 0.7):
        """
        Initialize the technical analysis agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            initial_cash: Starting cash amount
            risk_tolerance: Risk tolerance level (0-1)
            lookback_period: Number of periods to look back for analysis
            signal_threshold: Threshold for generating signals (0-1)
        """
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
    
    def calculate_sma(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, span: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=span, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using technical indicators.
        
        Args:
            market_data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            close_prices = symbol_data['close']
            
            # Calculate technical indicators
            results[symbol] = {
                'sma': self.calculate_sma(close_prices, self.lookback_period).iloc[-1],
                'ema': self.calculate_ema(close_prices, self.lookback_period).iloc[-1],
                'rsi': self.calculate_rsi(close_prices).iloc[-1],
                'macd': self.calculate_macd(close_prices)[0].iloc[-1],
                'macd_signal': self.calculate_macd(close_prices)[1].iloc[-1],
                'upper_band': self.calculate_bollinger_bands(close_prices)[0].iloc[-1],
                'lower_band': self.calculate_bollinger_bands(close_prices)[2].iloc[-1],
                'current_price': close_prices.iloc[-1],
                'price_sma_ratio': close_prices.iloc[-1] / self.calculate_sma(close_prices, self.lookback_period).iloc[-1] if not pd.isna(self.calculate_sma(close_prices, self.lookback_period).iloc[-1]) else 1.0
            }
            
            # Generate signals based on indicators
            signals = {}
            
            # RSI signals
            if results[symbol]['rsi'] < 30:
                signals['rsi'] = 'buy'  # Oversold
            elif results[symbol]['rsi'] > 70:
                signals['rsi'] = 'sell'  # Overbought
            else:
                signals['rsi'] = 'hold'
            
            # MACD signals
            if results[symbol]['macd'] > results[symbol]['macd_signal']:
                signals['macd'] = 'buy'  # Bullish crossover
            else:
                signals['macd'] = 'sell'  # Bearish crossover
            
            # Bollinger Bands signals
            if close_prices.iloc[-1] < results[symbol]['lower_band']:
                signals['bollinger'] = 'buy'  # Price below lower band
            elif close_prices.iloc[-1] > results[symbol]['upper_band']:
                signals['bollinger'] = 'sell'  # Price above upper band
            else:
                signals['bollinger'] = 'hold'
            
            # Price vs SMA
            if results[symbol]['price_sma_ratio'] > 1.05:
                signals['trend'] = 'sell'  # Price significantly above SMA
            elif results[symbol]['price_sma_ratio'] < 0.95:
                signals['trend'] = 'buy'  # Price significantly below SMA
            else:
                signals['trend'] = 'hold'
            
            results[symbol]['signals'] = signals
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        """
        Make trading decisions based on technical analysis.
        
        Args:
            market_data: DataFrame containing market data
            additional_info: Additional information that might be useful
            
        Returns:
            Dictionary containing trading decisions
        """
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            # Count buy and sell signals
            signals = data['signals']
            buy_count = sum(1 for signal in signals.values() if signal == 'buy')
            sell_count = sum(1 for signal in signals.values() if signal == 'sell')
            total_signals = len(signals)
            
            # Calculate signal strength
            buy_strength = buy_count / total_signals
            sell_strength = sell_count / total_signals
            
            # Make decision based on signal strength and threshold
            if buy_strength >= self.signal_threshold:
                # Determine position size based on risk tolerance and signal strength
                available_cash = self.cash * self.risk_tolerance * buy_strength
                price = data['current_price']
                quantity = int(available_cash / price) if price > 0 else 0
                
                decisions[symbol] = {
                    'action': 'buy',
                    'quantity': quantity,
                    'price': price,
                    'confidence': buy_strength
                }
            elif sell_strength >= self.signal_threshold:
                # Sell a portion of holdings based on signal strength
                if symbol in self.portfolio:
                    quantity = int(self.portfolio[symbol] * sell_strength * self.risk_tolerance)
                    if quantity > 0:
                        decisions[symbol] = {
                            'action': 'sell',
                            'quantity': quantity,
                            'price': data['current_price'],
                            'confidence': sell_strength
                        }
            else:
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': data['current_price'],
                    'confidence': max(buy_strength, sell_strength)
                }
        
        return decisions
