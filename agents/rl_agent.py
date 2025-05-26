import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class RLAgent(BaseAgent):
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        
        self.state_size = 10
        self.action_size = 3
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.q_table = {}
        
        self.features = [
            'price_change_1d',
            'price_change_5d',
            'volume_change',
            'rsi',
            'macd',
            'bollinger_position',
            'sentiment_score',
            'market_trend',
            'sector_performance',
            'portfolio_ratio'
        ]
    
    def _extract_features(self, market_data: pd.DataFrame, symbol: str) -> np.ndarray:
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            return np.zeros(self.state_size)
        
        if 'timestamp' in symbol_data.columns:
            sort_column = 'timestamp'
        elif 'date' in symbol_data.columns:
            sort_column = 'date'
        else:
            return np.zeros(self.state_size)
            
        symbol_data = symbol_data.sort_values(sort_column)
        
        try:
            latest_price = symbol_data['close'].iloc[-1]
            price_1d_ago = symbol_data['close'].iloc[-2] if len(symbol_data) > 1 else latest_price
            price_5d_ago = symbol_data['close'].iloc[-6] if len(symbol_data) > 5 else latest_price
            
            price_change_1d = (latest_price - price_1d_ago) / price_1d_ago
            price_change_5d = (latest_price - price_5d_ago) / price_5d_ago
            
            latest_volume = symbol_data['volume'].iloc[-1]
            prev_volume = symbol_data['volume'].iloc[-2] if len(symbol_data) > 1 else latest_volume
            volume_change = (latest_volume - prev_volume) / (prev_volume + 1)
            
            rsi = symbol_data['rsi'].iloc[-1] / 100.0 if 'rsi' in symbol_data.columns else 0.5
            macd = symbol_data['macd'].iloc[-1] if 'macd' in symbol_data.columns else 0.0
            
            macd_max = abs(macd) if abs(macd) > 0 else 1.0
            macd_normalized = macd / macd_max
            
            if all(col in symbol_data.columns for col in ['upper_band', 'lower_band']):
                upper = symbol_data['upper_band'].iloc[-1]
                lower = symbol_data['lower_band'].iloc[-1]
                middle = (upper + lower) / 2
                band_width = upper - lower
                bollinger_position = 2 * (latest_price - middle) / band_width if band_width > 0 else 0
                bollinger_position = max(-1, min(1, bollinger_position))
            else:
                bollinger_position = 0.0
            
            sentiment_score = symbol_data['sentiment_score'].iloc[-1] if 'sentiment_score' in symbol_data.columns else 0.5
            
            market_trend = 0.2
            sector_performance = 0.1
            
            portfolio_value = self.get_portfolio_value({symbol: latest_price})
            position_value = self.portfolio.get(symbol, 0) * latest_price
            portfolio_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            
            features = np.array([
                price_change_1d,
                price_change_5d,
                volume_change,
                rsi,
                macd_normalized,
                bollinger_position,
                sentiment_score,
                market_trend,
                sector_performance,
                portfolio_ratio
            ])
            
            features = np.nan_to_num(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for {symbol}: {e}")
            return np.zeros(self.state_size)
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        discretized = []
        for feature in state:
            if feature < -0.33:
                discretized.append(-1)
            elif feature > 0.33:
                discretized.append(1)
            else:
                discretized.append(0)
        
        return tuple(discretized)
    
    def _get_q_value(self, state: Tuple, action: int) -> float:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        return self.q_table[state][action]
    
    def _update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
        
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        if market_data.empty:
            return results
            
        if 'symbol' not in market_data.columns:
            return results
            
        for symbol in market_data['symbol'].unique():
            features = self._extract_features(market_data, symbol)
            
            symbol_data = market_data[market_data['symbol'] == symbol]
            current_price = symbol_data['close'].iloc[-1] if not symbol_data.empty else 0
            
            results[symbol] = {
                'features': features,
                'discretized_state': self._discretize_state(features),
                'current_price': current_price
            }
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            state = data['discretized_state']
            current_price = data['current_price']
            
            if np.random.rand() <= self.epsilon:
                action = np.random.randint(0, self.action_size)
            else:
                if state in self.q_table:
                    action = np.argmax(self.q_table[state])
                else:
                    self.q_table[state] = np.zeros(self.action_size)
                    action = np.random.randint(0, self.action_size)
            
            if action == 0:
                position_size = self.risk_tolerance * (0.5 + np.random.rand() * 0.5)
                available_cash = self.cash * position_size
                quantity = int(available_cash / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    decisions[symbol] = {
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'confidence': position_size
                    }
            elif action == 2:
                if symbol in self.portfolio:
                    position_size = self.risk_tolerance * (0.5 + np.random.rand() * 0.5)
                    quantity = int(self.portfolio[symbol] * position_size)
                    
                    if quantity > 0:
                        decisions[symbol] = {
                            'action': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'confidence': position_size
                        }
            else:
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': current_price,
                    'confidence': 0.5
                }
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return decisions
    
    def learn_from_experience(self, state, action, reward, next_state):
        action_idx = 0 if action == 'buy' else 2 if action == 'sell' else 1
        
        self._update_q_value(state, action_idx, reward, next_state)