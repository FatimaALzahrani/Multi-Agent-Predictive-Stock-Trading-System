"""
Reinforcement Learning Agent Module for Multi-Agent Trading System

This module implements a trading agent that uses reinforcement learning for decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class RLAgent(BaseAgent):
    """
    Trading agent that uses reinforcement learning to make decisions.
    
    Attributes:
        state_size (int): Size of the state representation
        action_size (int): Number of possible actions
        memory (List): Replay memory to store experiences
        gamma (float): Discount factor for future rewards
        epsilon (float): Exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for exploration
        learning_rate (float): Learning rate for the model
        model: The RL model (would be implemented with PyTorch or similar)
    """
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        """
        Initialize the reinforcement learning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            initial_cash: Starting cash amount
            risk_tolerance: Risk tolerance level (0-1)
        """
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        
        # RL parameters
        self.state_size = 10  # Size of state representation (features)
        self.action_size = 3  # Actions: Buy, Hold, Sell
        self.memory = []
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        # In a real implementation, we would initialize a neural network model here
        # For this prototype, we'll use a simplified approach
        self.q_table = {}  # Simple Q-table for demonstration
        
        # Features to extract from market data
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
        """
        Extract features from market data for the RL state representation.
        
        Args:
            market_data: DataFrame with market data
            symbol: Stock symbol to extract features for
            
        Returns:
            Array of normalized features
        """
        # In a real implementation, this would extract and normalize all features
        # For this prototype, we'll return a simplified feature vector
        
        symbol_data = market_data[market_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            # Return default features if no data
            return np.zeros(self.state_size)
        
        # Sort by timestamp or date to get time-series data
        # Check if 'timestamp' or 'date' column exists and use the appropriate one
        if 'timestamp' in symbol_data.columns:
            sort_column = 'timestamp'
        elif 'date' in symbol_data.columns:
            sort_column = 'date'
        else:
            # Return default features if neither column exists
            return np.zeros(self.state_size)
            
        symbol_data = symbol_data.sort_values(sort_column)
        
        # Extract basic features (simplified for prototype)
        try:
            # Price changes
            latest_price = symbol_data['close'].iloc[-1]
            price_1d_ago = symbol_data['close'].iloc[-2] if len(symbol_data) > 1 else latest_price
            price_5d_ago = symbol_data['close'].iloc[-6] if len(symbol_data) > 5 else latest_price
            
            price_change_1d = (latest_price - price_1d_ago) / price_1d_ago
            price_change_5d = (latest_price - price_5d_ago) / price_5d_ago
            
            # Volume change
            latest_volume = symbol_data['volume'].iloc[-1]
            prev_volume = symbol_data['volume'].iloc[-2] if len(symbol_data) > 1 else latest_volume
            volume_change = (latest_volume - prev_volume) / (prev_volume + 1)  # Avoid division by zero
            
            # Use available technical indicators if present
            rsi = symbol_data['rsi'].iloc[-1] / 100.0 if 'rsi' in symbol_data.columns else 0.5
            macd = symbol_data['macd'].iloc[-1] if 'macd' in symbol_data.columns else 0.0
            
            # Normalize MACD to [-1, 1] range
            macd_max = abs(macd) if abs(macd) > 0 else 1.0
            macd_normalized = macd / macd_max
            
            # Bollinger position (-1 to 1, where 0 is middle band)
            if all(col in symbol_data.columns for col in ['upper_band', 'lower_band']):
                upper = symbol_data['upper_band'].iloc[-1]
                lower = symbol_data['lower_band'].iloc[-1]
                middle = (upper + lower) / 2
                band_width = upper - lower
                bollinger_position = 2 * (latest_price - middle) / band_width if band_width > 0 else 0
                bollinger_position = max(-1, min(1, bollinger_position))  # Clamp to [-1, 1]
            else:
                bollinger_position = 0.0
            
            # Sentiment score if available
            sentiment_score = symbol_data['sentiment_score'].iloc[-1] if 'sentiment_score' in symbol_data.columns else 0.5
            
            # Market and sector trends (simplified)
            market_trend = 0.2  # Placeholder: positive market trend
            sector_performance = 0.1  # Placeholder: slightly positive sector performance
            
            # Portfolio ratio (position size relative to portfolio)
            portfolio_value = self.get_portfolio_value({symbol: latest_price})
            position_value = self.portfolio.get(symbol, 0) * latest_price
            portfolio_ratio = position_value / portfolio_value if portfolio_value > 0 else 0
            
            # Combine features
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
            
            # Replace NaN values with 0
            features = np.nan_to_num(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for {symbol}: {e}")
            return np.zeros(self.state_size)
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """
        Discretize continuous state for Q-table lookup.
        
        Args:
            state: Continuous state vector
            
        Returns:
            Tuple representation of discretized state
        """
        # Discretize each feature into 3 bins: low, medium, high
        discretized = []
        for feature in state:
            if feature < -0.33:
                discretized.append(-1)  # Low
            elif feature > 0.33:
                discretized.append(1)   # High
            else:
                discretized.append(0)   # Medium
        
        return tuple(discretized)
    
    def _get_q_value(self, state: Tuple, action: int) -> float:
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: Discretized state tuple
            action: Action index
            
        Returns:
            Q-value
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        return self.q_table[state][action]
    
    def _update_q_value(self, state: Tuple, action: int, reward: float, next_state: Tuple):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)
        
        # Q-learning update rule
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        
        self.q_table[state][action] = new_q
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using RL features.
        
        Args:
            market_data: DataFrame with market data
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        
        # Check if market_data is empty
        if market_data.empty:
            return results
            
        # Check if 'symbol' column exists
        if 'symbol' not in market_data.columns:
            return results
            
        for symbol in market_data['symbol'].unique():
            # Extract features for the symbol
            features = self._extract_features(market_data, symbol)
            
            # Get current price
            symbol_data = market_data[market_data['symbol'] == symbol]
            current_price = symbol_data['close'].iloc[-1] if not symbol_data.empty else 0
            
            # Store results
            results[symbol] = {
                'features': features,
                'discretized_state': self._discretize_state(features),
                'current_price': current_price
            }
        
        return results
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        """
        Make trading decisions using reinforcement learning.
        
        Args:
            market_data: DataFrame containing market data
            additional_info: Additional information that might be useful
            
        Returns:
            Dictionary containing trading decisions
        """
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            state = data['discretized_state']
            current_price = data['current_price']
            
            # Epsilon-greedy action selection
            if np.random.rand() <= self.epsilon:
                # Exploration: random action
                action = np.random.randint(0, self.action_size)
            else:
                # Exploitation: best known action
                if state in self.q_table:
                    action = np.argmax(self.q_table[state])
                else:
                    # If state not seen before, initialize and choose random action
                    self.q_table[state] = np.zeros(self.action_size)
                    action = np.random.randint(0, self.action_size)
            
            # Convert action to trading decision
            if action == 0:  # Buy
                # Calculate position size based on risk tolerance
                position_size = self.risk_tolerance * (0.5 + np.random.rand() * 0.5)  # Add some randomness
                available_cash = self.cash * position_size
                quantity = int(available_cash / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    decisions[symbol] = {
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'confidence': position_size
                    }
            elif action == 2:  # Sell
                if symbol in self.portfolio:
                    # Sell a portion of holdings
                    position_size = self.risk_tolerance * (0.5 + np.random.rand() * 0.5)
                    quantity = int(self.portfolio[symbol] * position_size)
                    
                    if quantity > 0:
                        decisions[symbol] = {
                            'action': 'sell',
                            'quantity': quantity,
                            'price': current_price,
                            'confidence': position_size
                        }
            else:  # Hold (action == 1)
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': current_price,
                    'confidence': 0.5
                }
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return decisions
    
    def learn_from_experience(self, state, action, reward, next_state):
        """
        Update the agent's knowledge based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Convert action string to index
        action_idx = 0 if action == 'buy' else 2 if action == 'sell' else 1
        
        # Update Q-value
        self._update_q_value(state, action_idx, reward, next_state)
