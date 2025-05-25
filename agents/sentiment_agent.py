"""
Sentiment Analysis Agent Module for Multi-Agent Trading System

This module implements a trading agent that uses sentiment analysis for decision making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """
    Trading agent that uses sentiment analysis to make decisions.
    
    Attributes:
        sentiment_threshold_buy (float): Threshold for positive sentiment to trigger buy
        sentiment_threshold_sell (float): Threshold for negative sentiment to trigger sell
        sentiment_lookback (int): Number of periods to consider for sentiment trend
        news_weight (float): Weight given to news sentiment vs social media
        social_weight (float): Weight given to social media sentiment vs news
    """
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        """
        Initialize the sentiment analysis agent.
        
        Args:
            agent_id: Unique identifier for the agent
            name: Human-readable name for the agent
            initial_cash: Starting cash amount
            risk_tolerance: Risk tolerance level (0-1)
        """
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.sentiment_threshold_buy = 0.6  # Positive sentiment threshold
        self.sentiment_threshold_sell = 0.4  # Negative sentiment threshold
        self.sentiment_lookback = 3  # Days to look back for sentiment trend
        self.news_weight = 0.7  # News sentiment weight (70%)
        self.social_weight = 0.3  # Social media sentiment weight (30%)
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market data using sentiment metrics.
        
        Args:
            market_data: DataFrame with sentiment data for stocks
            
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
            
        # Group by symbol to analyze each stock
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            # Sort by date to get the most recent data first
            # Check if 'timestamp' or 'date' column exists and use the appropriate one
            if 'timestamp' in symbol_data.columns:
                sort_column = 'timestamp'
            elif 'date' in symbol_data.columns:
                sort_column = 'date'
            else:
                # Skip if neither column exists
                continue
                
            symbol_data = symbol_data.sort_values(sort_column, ascending=False)
            
            # Get recent sentiment data (limited by lookback period)
            recent_data = symbol_data.head(self.sentiment_lookback)
            
            if not recent_data.empty:
                # Extract sentiment metrics
                try:
                    # Check if sentiment columns exist, otherwise use dummy values
                    if 'news_sentiment' in recent_data.columns and 'social_sentiment' in recent_data.columns:
                        # Calculate average sentiment scores
                        avg_news_sentiment = recent_data['news_sentiment'].mean()
                        avg_social_sentiment = recent_data['social_sentiment'].mean()
                    elif 'sentiment_score' in recent_data.columns:
                        # Use sentiment_score for both if it exists
                        avg_news_sentiment = recent_data['sentiment_score'].mean()
                        avg_social_sentiment = recent_data['sentiment_score'].mean()
                    else:
                        # Generate random sentiment scores if none exist
                        avg_news_sentiment = np.random.uniform(0.3, 0.7)
                        avg_social_sentiment = np.random.uniform(0.3, 0.7)
                    
                    # Calculate weighted sentiment score
                    weighted_sentiment = (
                        avg_news_sentiment * self.news_weight +
                        avg_social_sentiment * self.social_weight
                    )
                    
                    # Calculate sentiment trend (positive value means improving sentiment)
                    if len(recent_data) > 1:
                        if 'news_sentiment' in recent_data.columns and 'social_sentiment' in recent_data.columns:
                            oldest_sentiment = (
                                recent_data.iloc[-1]['news_sentiment'] * self.news_weight +
                                recent_data.iloc[-1]['social_sentiment'] * self.social_weight
                            )
                            newest_sentiment = (
                                recent_data.iloc[0]['news_sentiment'] * self.news_weight +
                                recent_data.iloc[0]['social_sentiment'] * self.social_weight
                            )
                        elif 'sentiment_score' in recent_data.columns:
                            oldest_sentiment = recent_data.iloc[-1]['sentiment_score']
                            newest_sentiment = recent_data.iloc[0]['sentiment_score']
                        else:
                            # Random trend if no sentiment data
                            oldest_sentiment = np.random.uniform(0.3, 0.7)
                            newest_sentiment = np.random.uniform(0.3, 0.7)
                            
                        sentiment_trend = newest_sentiment - oldest_sentiment
                    else:
                        sentiment_trend = 0.0
                    
                    # Get current price
                    current_price = recent_data.iloc[0]['close'] if 'close' in recent_data.columns else 0.0
                    
                    # Store results
                    results[symbol] = {
                        'news_sentiment': avg_news_sentiment,
                        'social_sentiment': avg_social_sentiment,
                        'weighted_sentiment': weighted_sentiment,
                        'sentiment_trend': sentiment_trend,
                        'current_price': current_price,
                        'sentiment_signal': self._get_sentiment_signal(weighted_sentiment, sentiment_trend)
                    }
                    
                except Exception as e:
                    # Handle missing data
                    results[symbol] = {
                        'error': str(e),
                        'weighted_sentiment': 0.5,  # Neutral sentiment as default
                        'sentiment_trend': 0.0
                    }
        
        return results
    
    def _get_sentiment_signal(self, sentiment: float, trend: float) -> str:
        """
        Determine trading signal based on sentiment and trend.
        
        Args:
            sentiment: Weighted sentiment score
            trend: Sentiment trend (change)
            
        Returns:
            Signal string ('strong_buy', 'buy', 'neutral', 'sell', 'strong_sell')
        """
        # Base signal on sentiment level
        if sentiment > self.sentiment_threshold_buy:
            base_signal = 'buy'
        elif sentiment < self.sentiment_threshold_sell:
            base_signal = 'sell'
        else:
            return 'neutral'  # Neutral zone
        
        # Strengthen or weaken signal based on trend
        trend_threshold = 0.1  # Threshold for significant trend
        
        if base_signal == 'buy' and trend > trend_threshold:
            return 'strong_buy'  # Strong positive sentiment with improving trend
        elif base_signal == 'sell' and trend < -trend_threshold:
            return 'strong_sell'  # Strong negative sentiment with worsening trend
        
        return base_signal
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        """
        Make trading decisions based on sentiment analysis.
        
        Args:
            market_data: DataFrame containing market data with sentiment metrics
            additional_info: Additional information that might be useful
            
        Returns:
            Dictionary containing trading decisions
        """
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            if 'error' in data:
                continue  # Skip stocks with missing data
            
            sentiment_signal = data.get('sentiment_signal', 'neutral')
            current_price = data.get('current_price', 0)
            
            # Make decision based on sentiment signal
            if sentiment_signal in ('buy', 'strong_buy'):
                # Calculate position size based on sentiment and risk tolerance
                confidence = 0.7 if sentiment_signal == 'buy' else 0.9
                position_size = self.risk_tolerance * confidence
                available_cash = self.cash * position_size
                quantity = int(available_cash / current_price) if current_price > 0 else 0
                
                if quantity > 0:
                    decisions[symbol] = {
                        'action': 'buy',
                        'quantity': quantity,
                        'price': current_price,
                        'confidence': confidence
                    }
            elif sentiment_signal in ('sell', 'strong_sell'):
                # Sell based on negative sentiment
                if symbol in self.portfolio:
                    confidence = 0.7 if sentiment_signal == 'sell' else 0.9
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
