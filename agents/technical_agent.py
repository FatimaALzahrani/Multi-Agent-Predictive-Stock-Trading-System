import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class TechnicalAgent(BaseAgent):

    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5, lookback_period: int = 20,
                 signal_threshold: float = 0.7):

        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.indicators = ['sma', 'ema', 'rsi', 'macd', 'bollinger']
        self.lookback_period = lookback_period
        self.signal_threshold = signal_threshold
    
    def calculate_sma(self, prices: pd.Series, window: int = 20) -> pd.Series:
        return prices.rolling(window=window).mean()
    
    def calculate_ema(self, prices: pd.Series, span: int = 20) -> pd.Series:
        return prices.ewm(span=span, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        fast_ema = prices.ewm(span=fast, adjust=False).mean()
        slow_ema = prices.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        sma = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            close_prices = symbol_data['close']
            
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
            
            signals = {}
            
            if results[symbol]['rsi'] < 30:
                signals['rsi'] = 'buy' 
            elif results[symbol]['rsi'] > 70:
                signals['rsi'] = 'sell'  
            else:
                signals['rsi'] = 'hold'
            
            if results[symbol]['macd'] > results[symbol]['macd_signal']:
                signals['macd'] = 'buy' 
            else:
                signals['macd'] = 'sell' 
            
            if close_prices.iloc[-1] < results[symbol]['lower_band']:
                signals['bollinger'] = 'buy'
            elif close_prices.iloc[-1] > results[symbol]['upper_band']:
                signals['bollinger'] = 'sell'
            else:
                signals['bollinger'] = 'hold'
            
            if results[symbol]['price_sma_ratio'] > 1.05:
                signals['trend'] = 'sell'  
            elif results[symbol]['price_sma_ratio'] < 0.95:
                signals['trend'] = 'buy'  
            else:
                signals['trend'] = 'hold'
            
            results[symbol]['signals'] = signals
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            signals = data['signals']
            buy_count = sum(1 for signal in signals.values() if signal == 'buy')
            sell_count = sum(1 for signal in signals.values() if signal == 'sell')
            total_signals = len(signals)
            
            buy_strength = buy_count / total_signals
            sell_strength = sell_count / total_signals
            
            if buy_strength >= self.signal_threshold:
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
