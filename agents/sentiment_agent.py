import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    
    def __init__(self, agent_id: str, name: str, initial_cash: float = 100000.0,
                 risk_tolerance: float = 0.5):
        super().__init__(agent_id, name, initial_cash, risk_tolerance)
        self.sentiment_threshold_buy = 0.6
        self.sentiment_threshold_sell = 0.4
        self.sentiment_lookback = 3
        self.news_weight = 0.7
        self.social_weight = 0.3
    
    def analyze_market(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        results = {}
        
        if market_data.empty:
            return results
            
        if 'symbol' not in market_data.columns:
            return results
            
        for symbol in market_data['symbol'].unique():
            symbol_data = market_data[market_data['symbol'] == symbol].copy()
            
            if 'timestamp' in symbol_data.columns:
                sort_column = 'timestamp'
            elif 'date' in symbol_data.columns:
                sort_column = 'date'
            else:
                continue
                
            symbol_data = symbol_data.sort_values(sort_column, ascending=False)
            
            recent_data = symbol_data.head(self.sentiment_lookback)
            
            if not recent_data.empty:
                try:
                    if 'news_sentiment' in recent_data.columns and 'social_sentiment' in recent_data.columns:
                        avg_news_sentiment = recent_data['news_sentiment'].mean()
                        avg_social_sentiment = recent_data['social_sentiment'].mean()
                    elif 'sentiment_score' in recent_data.columns:
                        avg_news_sentiment = recent_data['sentiment_score'].mean()
                        avg_social_sentiment = recent_data['sentiment_score'].mean()
                    else:
                        avg_news_sentiment = np.random.uniform(0.3, 0.7)
                        avg_social_sentiment = np.random.uniform(0.3, 0.7)
                    
                    weighted_sentiment = (
                        avg_news_sentiment * self.news_weight +
                        avg_social_sentiment * self.social_weight
                    )
                    
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
                            oldest_sentiment = np.random.uniform(0.3, 0.7)
                            newest_sentiment = np.random.uniform(0.3, 0.7)
                            
                        sentiment_trend = newest_sentiment - oldest_sentiment
                    else:
                        sentiment_trend = 0.0
                    
                    current_price = recent_data.iloc[0]['close'] if 'close' in recent_data.columns else 0.0
                    
                    results[symbol] = {
                        'news_sentiment': avg_news_sentiment,
                        'social_sentiment': avg_social_sentiment,
                        'weighted_sentiment': weighted_sentiment,
                        'sentiment_trend': sentiment_trend,
                        'current_price': current_price,
                        'sentiment_signal': self._get_sentiment_signal(weighted_sentiment, sentiment_trend)
                    }
                    
                except Exception as e:
                    results[symbol] = {
                        'error': str(e),
                        'weighted_sentiment': 0.5,
                        'sentiment_trend': 0.0
                    }
        
        return results
    
    def _get_sentiment_signal(self, sentiment: float, trend: float) -> str:
        if sentiment > self.sentiment_threshold_buy:
            base_signal = 'buy'
        elif sentiment < self.sentiment_threshold_sell:
            base_signal = 'sell'
        else:
            return 'neutral'
        
        trend_threshold = 0.1
        
        if base_signal == 'buy' and trend > trend_threshold:
            return 'strong_buy'
        elif base_signal == 'sell' and trend < -trend_threshold:
            return 'strong_sell'
        
        return base_signal
    
    def make_decision(self, market_data: pd.DataFrame, additional_info: Dict = None) -> Dict[str, Any]:
        analysis = self.analyze_market(market_data)
        decisions = {}
        
        for symbol, data in analysis.items():
            if 'error' in data:
                continue
            
            sentiment_signal = data.get('sentiment_signal', 'neutral')
            current_price = data.get('current_price', 0)
            
            if sentiment_signal in ('buy', 'strong_buy'):
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
                decisions[symbol] = {
                    'action': 'hold',
                    'quantity': 0,
                    'price': current_price,
                    'confidence': 0.5
                }
        
        return decisions