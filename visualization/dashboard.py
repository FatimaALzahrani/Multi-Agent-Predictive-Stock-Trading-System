import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class TradingDashboard:
    
    def __init__(self, market, port=8050):
        self.market = market
        self.port = port
        self.simulation_running = False
        
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Multi-Agent Trading System", className="text-center mb-4"),
                    html.Hr()
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Simulation Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Simulation Speed:"),
                                    dcc.Slider(
                                        id="simulation-speed",
                                        min=1,
                                        max=10,
                                        step=1,
                                        value=5,
                                        marks={i: str(i) for i in range(1, 11)}
                                    )
                                ], width=6),
                                dbc.Col([
                                    html.Div([
                                        dbc.Button("Start", id="start-button", color="success", className="me-2"),
                                        dbc.Button("Pause", id="pause-button", color="warning", className="me-2"),
                                        dbc.Button("Reset", id="reset-button", color="danger")
                                    ], className="d-flex justify-content-center")
                                ], width=6)
                            ]),
                            html.Div(id="simulation-status", className="mt-3 text-center")
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Overview"),
                        dbc.CardBody([
                            dcc.Graph(id="market-graph", style={"height": "400px"})
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Agent Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="performance-graph", style={"height": "400px"})
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Agent Actions"),
                        dbc.CardBody([
                            dcc.Graph(id="agent-actions-graph", style={"height": "300px"})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Composition"),
                        dbc.CardBody([
                            dcc.Graph(id="portfolio-graph", style={"height": "300px"})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Agent Details"),
                        dbc.CardBody([
                            dbc.Tabs([
                                dbc.Tab(label="Technical Agent", tab_id="tab-technical"),
                                dbc.Tab(label="Fundamental Agent", tab_id="tab-fundamental"),
                                dbc.Tab(label="Sentiment Agent", tab_id="tab-sentiment"),
                                dbc.Tab(label="RL Agent", tab_id="tab-rl")
                            ], id="agent-tabs", active_tab="tab-technical"),
                            html.Div(id="agent-details-content", className="mt-3")
                        ])
                    ])
                ], width=12)
            ]),
            
            dcc.Store(id="market-data-store", data={}),
            dcc.Store(id="agent-data-store", data={}),
            dcc.Store(id="simulation-state", data={"running": False, "step": 0}),
            dcc.Interval(id="simulation-interval", interval=1000, n_intervals=0, disabled=True)
        ], fluid=True, className="mt-4")
    
    def _setup_callbacks(self):
        
        @self.app.callback(
            [Output("simulation-interval", "disabled"),
             Output("simulation-status", "children"),
             Output("simulation-state", "data")],
            [Input("start-button", "n_clicks"),
             Input("pause-button", "n_clicks"),
             Input("reset-button", "n_clicks")],
            [State("simulation-interval", "disabled"),
             State("simulation-state", "data")],
            prevent_initial_call=True
        )
        def control_simulation(start_clicks, pause_clicks, reset_clicks, is_disabled, sim_state):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_disabled, "Simulation not started", sim_state
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-button":
                new_state = sim_state.copy()
                new_state["running"] = True
                return False, "Simulation running", new_state
            elif button_id == "pause-button":
                new_state = sim_state.copy()
                new_state["running"] = False
                return True, "Simulation paused", new_state
            elif button_id == "reset-button":
                try:
                    self.market.reset()
                    new_state = {"running": False, "step": 0}
                    return True, "Simulation reset", new_state
                except Exception as e:
                    return True, f"Reset failed: {str(e)}", sim_state
            
            return is_disabled, "Simulation status unknown", sim_state
        
        @self.app.callback(
            Output("simulation-interval", "interval"),
            Input("simulation-speed", "value")
        )
        def update_interval(speed):
            return max(200, 2200 - (speed * 200))
        
        @self.app.callback(
            [Output("market-data-store", "data"),
             Output("agent-data-store", "data")],
            [Input("simulation-interval", "n_intervals")],
            [State("simulation-state", "data"),
             State("market-data-store", "data"),
             State("agent-data-store", "data")],
            prevent_initial_call=True
        )
        def update_simulation(n_intervals, sim_state, current_market_data, current_agent_data):
            if not sim_state.get("running", False):
                return current_market_data or {}, current_agent_data or {}
            
            try:
                step_result = self.market.step()
                agent_performance = self.market.get_agent_performance()
                
                if step_result:
                    step_result["step"] = n_intervals
                
                return step_result or {}, agent_performance or {}
                
            except Exception as e:
                print(f"Error in simulation step: {e}")
                return current_market_data or {}, current_agent_data or {}
        
        @self.app.callback(
            Output("market-graph", "figure"),
            Input("market-data-store", "data"),
            prevent_initial_call=True
        )
        def update_market_graph(market_data):
            fig = go.Figure()
            
            try:
                historical_data = self.market.get_historical_window(30)
                
                if historical_data is None or historical_data.empty:
                    fig.update_layout(
                        title="No market data available",
                        template="plotly_dark",
                        xaxis_title="Date",
                        yaxis_title="Price"
                    )
                    return fig
                
                if 'symbol' not in historical_data.columns:
                    fig.update_layout(
                        title="Invalid market data format",
                        template="plotly_dark"
                    )
                    return fig
                
                for symbol in self.market.symbols:
                    symbol_data = historical_data[historical_data['symbol'] == symbol]
                    
                    if not symbol_data.empty and 'close' in symbol_data.columns:
                        fig.add_trace(go.Scatter(
                            x=symbol_data.get('date', range(len(symbol_data))),
                            y=symbol_data['close'],
                            mode='lines',
                            name=symbol,
                            line=dict(width=2)
                        ))
                
                fig.update_layout(
                    title="Stock Prices",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode='x unified'
                )
                
            except Exception as e:
                print(f"Error updating market graph: {e}")
                fig.update_layout(
                    title=f"Error loading market data: {str(e)}",
                    template="plotly_dark"
                )
            
            return fig
        
        @self.app.callback(
            Output("performance-graph", "figure"),
            Input("agent-data-store", "data"),
            prevent_initial_call=True
        )
        def update_performance_graph(agent_data):
            fig = go.Figure()
            
            try:
                if not agent_data:
                    fig.update_layout(
                        title="No agent performance data",
                        template="plotly_dark"
                    )
                    return fig
                
                agents = list(agent_data.keys())
                returns = []
                
                for agent in agents:
                    agent_info = agent_data[agent]
                    if isinstance(agent_info, dict) and 'total_return' in agent_info:
                        returns.append(agent_info['total_return'] * 100)
                    else:
                        returns.append(0)
                
                colors = ['green' if r >= 0 else 'red' for r in returns]
                
                fig.add_trace(go.Bar(
                    x=agents,
                    y=returns,
                    text=[f"{ret:.2f}%" for ret in returns],
                    textposition='auto',
                    marker_color=colors
                ))
                
                fig.update_layout(
                    title="Agent Performance (Return %)",
                    xaxis_title="Agent",
                    yaxis_title="Return (%)",
                    template="plotly_dark"
                )
                
            except Exception as e:
                print(f"Error updating performance graph: {e}")
                fig.update_layout(
                    title=f"Error loading performance data: {str(e)}",
                    template="plotly_dark"
                )
            
            return fig
        
        @self.app.callback(
            Output("agent-actions-graph", "figure"),
            Input("market-data-store", "data"),
            prevent_initial_call=True
        )
        def update_actions_graph(market_data):
            fig = go.Figure()
            
            try:
                if not hasattr(self.market, 'market_history') or not self.market.market_history:
                    fig.update_layout(
                        title="No action history available",
                        template="plotly_dark"
                    )
                    return fig
                
                recent_history = self.market.market_history[-10:] if len(self.market.market_history) >= 10 else self.market.market_history
                
                agent_ids = [agent.agent_id for agent in self.market.agents]
                buy_counts = {agent_id: 0 for agent_id in agent_ids}
                sell_counts = {agent_id: 0 for agent_id in agent_ids}
                
                for state in recent_history:
                    if 'agent_actions' in state:
                        for agent_id, actions in state['agent_actions'].items():
                            if isinstance(actions, dict):
                                for symbol, action_data in actions.items():
                                    if isinstance(action_data, dict) and 'action' in action_data:
                                        if action_data['action'] == 'buy':
                                            buy_counts[agent_id] += 1
                                        elif action_data['action'] == 'sell':
                                            sell_counts[agent_id] += 1
                
                fig.add_trace(go.Bar(
                    x=agent_ids,
                    y=[buy_counts[agent_id] for agent_id in agent_ids],
                    name='Buy',
                    marker_color='green'
                ))
                
                fig.add_trace(go.Bar(
                    x=agent_ids,
                    y=[sell_counts[agent_id] for agent_id in agent_ids],
                    name='Sell',
                    marker_color='red'
                ))
                
                fig.update_layout(
                    title="Recent Agent Actions (Last 10 Steps)",
                    xaxis_title="Agent",
                    yaxis_title="Action Count",
                    barmode='group',
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
            except Exception as e:
                print(f"Error updating actions graph: {e}")
                fig.update_layout(
                    title=f"Error loading action data: {str(e)}",
                    template="plotly_dark"
                )
            
            return fig
        
        @self.app.callback(
            Output("portfolio-graph", "figure"),
            Input("agent-data-store", "data"),
            prevent_initial_call=True
        )
        def update_portfolio_graph(agent_data):
            fig = go.Figure()
            
            try:
                if not self.market.agents:
                    fig.update_layout(
                        title="No agents available",
                        template="plotly_dark"
                    )
                    return fig
                
                agent = self.market.agents[0]
                
                labels = []
                values = []
                
                if hasattr(agent, 'portfolio') and agent.portfolio:
                    for symbol, quantity in agent.portfolio.items():
                        if quantity > 0:
                            current_price = getattr(self.market, 'current_prices', {}).get(symbol, 0)
                            value = quantity * current_price
                            if value > 0:
                                labels.append(f"{symbol} ({quantity} shares)")
                                values.append(value)
                
                if hasattr(agent, 'cash') and agent.cash > 0:
                    labels.append('Cash')
                    values.append(agent.cash)
                
                if not labels:
                    labels = ['No Holdings']
                    values = [1]
                
                fig.add_trace(go.Pie(
                    labels=labels,
                    values=values,
                    hole=.3,
                    textinfo='label+percent',
                    textposition='auto'
                ))
                
                fig.update_layout(
                    title=f"Portfolio Composition - {getattr(agent, 'name', 'Unknown Agent')}",
                    template="plotly_dark"
                )
                
            except Exception as e:
                print(f"Error updating portfolio graph: {e}")
                fig.update_layout(
                    title=f"Error loading portfolio data: {str(e)}",
                    template="plotly_dark"
                )
            
            return fig
        
        @self.app.callback(
            Output("agent-details-content", "children"),
            [Input("agent-tabs", "active_tab")],
            [State("agent-data-store", "data")],
            prevent_initial_call=True
        )
        def update_agent_details(active_tab, agent_data):
            try:
                if not active_tab:
                    return html.P("Select an agent tab to view details")
                
                agent_type = active_tab.split("-")[1] if "-" in active_tab else ""
                
                target_agent = None
                for agent in self.market.agents:
                    if agent_type.lower() in agent.agent_id.lower():
                        target_agent = agent
                        break
                
                if not target_agent:
                    return html.P(f"No {agent_type} agent found")
                
                try:
                    metrics = getattr(target_agent, 'performance_metrics', {})
                except:
                    metrics = {}
                
                portfolio_value = 0
                try:
                    if hasattr(target_agent, 'portfolio') and target_agent.portfolio:
                        current_prices = getattr(self.market, 'current_prices', {})
                        portfolio_value = sum([
                            target_agent.portfolio.get(symbol, 0) * current_prices.get(symbol, 0)
                            for symbol in target_agent.portfolio
                        ])
                except:
                    portfolio_value = 0
                
                details = [
                    html.H4(getattr(target_agent, 'name', 'Unknown Agent')),
                    html.P(f"Agent ID: {target_agent.agent_id}"),
                    html.P(f"Cash: ${getattr(target_agent, 'cash', 0):.2f}"),
                    html.P(f"Portfolio Value: ${portfolio_value:.2f}"),
                    html.P(f"Total Value: ${getattr(target_agent, 'cash', 0) + portfolio_value:.2f}")
                ]
                
                if metrics:
                    details.extend([
                        html.P(f"Total Return: {metrics.get('total_return', 0) * 100:.2f}%"),
                        html.P(f"Trades Count: {metrics.get('trades_count', 0)}")
                    ])
                
                details.append(html.H5("Recent Transactions", className="mt-3"))
                
                try:
                    if hasattr(target_agent, 'transaction_history') and target_agent.transaction_history:
                        recent_transactions = target_agent.transaction_history[-5:]
                        if recent_transactions:
                            df = pd.DataFrame(recent_transactions)
                            details.append(
                                dbc.Table.from_dataframe(
                                    df,
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm"
                                )
                            )
                        else:
                            details.append(html.P("No recent transactions"))
                    else:
                        details.append(html.P("No transaction history available"))
                except Exception as e:
                    details.append(html.P(f"Error loading transactions: {str(e)}"))
                
                return html.Div(details)
                
            except Exception as e:
                return html.P(f"Error loading agent details: {str(e)}")
    
    def run(self, debug=False, host='127.0.0.1'):
        try:
            self.app.run(debug=debug, host=host, port=self.port)
        except Exception as e:
            print(f"Error starting dashboard: {e}")
            try:
                self.app.run_server(debug=debug, host=host, port=self.port)
            except Exception as e2:
                print(f"Error with fallback method: {e2}")
                raise e