import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import datetime


class MockMarketEnvironment:
    
    def __init__(self, symbols: List[str], start_date: str, end_date: str = None):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.now().strftime('%Y-%m-%d')
        self.current_step = 0
        self.data = None
        self.current_prices = {}
        self.agents = []
        self.daily_returns = {}
        self.market_history = []
        
        self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        all_data = []
        
        start_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        end_date = datetime.datetime.strptime(self.end_date, '%Y-%m-%d')
        
        days = (end_date - start_date).days + 1
        date_range = [start_date + datetime.timedelta(days=i) for i in range(days)]
        
        for symbol in self.symbols:
            if symbol == 'AAPL':
                initial_price = 150.0
            elif symbol == 'MSFT':
                initial_price = 300.0
            elif symbol == 'GOOGL':
                initial_price = 2500.0
            elif symbol == 'AMZN':
                initial_price = 3000.0
            elif symbol == 'TSLA':
                initial_price = 700.0
            else:
                initial_price = 100.0
            
            prices = [initial_price]
            for i in range(1, days):
                daily_return = np.random.normal(0.0005, 0.015)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
            
            symbol_data = pd.DataFrame({
                'date': date_range,
                'symbol': symbol,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
                'close': prices,
                'adj close': prices,
                'volume': [int(np.random.uniform(1000000, 10000000)) for _ in range(days)]
            })
            
            all_data.append(symbol_data)
        
        self.data = pd.concat(all_data, ignore_index=True)
        self._calculate_features()
    
    def _calculate_features(self):
        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            
            if not symbol_data.empty:
                symbol_data = symbol_data.sort_values('date')
                
                symbol_data['daily_return'] = symbol_data['close'].pct_change()
                symbol_data['sma_20'] = symbol_data['close'].rolling(window=20).mean()
                symbol_data['ema_20'] = symbol_data['close'].ewm(span=20, adjust=False).mean()
                
                delta = symbol_data['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                symbol_data['rsi'] = 100 - (100 / (1 + rs))
                
                ema_12 = symbol_data['close'].ewm(span=12, adjust=False).mean()
                ema_26 = symbol_data['close'].ewm(span=26, adjust=False).mean()
                symbol_data['macd'] = ema_12 - ema_26
                symbol_data['macd_signal'] = symbol_data['macd'].ewm(span=9, adjust=False).mean()
                
                sma_20 = symbol_data['close'].rolling(window=20).mean()
                std_20 = symbol_data['close'].rolling(window=20).std()
                symbol_data['upper_band'] = sma_20 + (std_20 * 2)
                symbol_data['lower_band'] = sma_20 - (std_20 * 2)
                
                symbol_data['sentiment_score'] = np.random.uniform(0.3, 0.7, len(symbol_data))
                
                self.data.update(symbol_data)
    
    def add_agent(self, agent):
        self.agents.append(agent)
    
    def get_current_data(self) -> pd.DataFrame:
        dates = sorted(self.data['date'].unique())
        
        if self.current_step < len(dates):
            current_date = dates[self.current_step]
            current_data = self.data[self.data['date'] == current_date].copy()
            
            for symbol in self.symbols:
                symbol_data = current_data[current_data['symbol'] == symbol]
                if not symbol_data.empty:
                    self.current_prices[symbol] = symbol_data['close'].iloc[0]
            
            return current_data
        else:
            return pd.DataFrame()
    
    def get_historical_window(self, window_size: int = 30) -> pd.DataFrame:
        dates = sorted(self.data['date'].unique())
        
        if self.current_step < len(dates):
            start_idx = max(0, self.current_step - window_size + 1)
            date_range = dates[start_idx:self.current_step + 1]
            historical_data = self.data[self.data['date'].isin(date_range)].copy()
            
            return historical_data
        else:
            return pd.DataFrame()
    
    def step(self) -> Dict[str, Any]:
        current_data = self.get_current_data()
        
        if current_data.empty:
            return {'done': True}
        
        historical_data = self.get_historical_window()
        
        market_state = {
            'date': current_data['date'].iloc[0],
            'prices': self.current_prices.copy(),
            'agent_actions': {}
        }
        
        for agent in self.agents:
            analysis = agent.analyze_market(historical_data)
            decisions = agent.make_decision(historical_data)
            
            agent_actions = {}
            for symbol, decision in decisions.items():
                if decision['action'] in ['buy', 'sell'] and decision['quantity'] > 0:
                    trade_result = agent.execute_trade(
                        symbol=symbol,
                        quantity=decision['quantity'] if decision['action'] == 'buy' else -decision['quantity'],
                        price=self.current_prices.get(symbol, 0)
                    )
                    agent_actions[symbol] = {
                        'action': decision['action'],
                        'quantity': decision['quantity'],
                        'price': self.current_prices.get(symbol, 0),
                        'result': trade_result
                    }
            
            market_state['agent_actions'][agent.agent_id] = agent_actions
            agent.calculate_returns(self.current_prices, 100000)
        
        self.market_history.append(market_state)
        self.current_step += 1
        
        return {
            'done': False,
            'current_step': self.current_step,
            'date': current_data['date'].iloc[0],
            'prices': self.current_prices.copy(),
            'agent_actions': market_state['agent_actions']
        }
    
    def run_simulation(self, steps: int = None) -> List[Dict[str, Any]]:
        results = []
        done = False
        step_count = 0
        
        while not done and (steps is None or step_count < steps):
            step_result = self.step()
            results.append(step_result)
            done = step_result['done']
            step_count += 1
        
        return results
    
    def get_agent_performance(self) -> Dict[str, Dict[str, float]]:
        performance = {}
        
        for agent in self.agents:
            performance[agent.agent_id] = agent.performance_metrics
        
        return performance
    
    def reset(self):
        self.current_step = 0
        self.market_history = []
        
        for agent in self.agents:
            agent.reset()