"""
Dashboard Module for Multi-Agent Trading System

This module implements an interactive dashboard for visualizing the trading system.
"""

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any


class TradingDashboard:
    """
    Interactive dashboard for visualizing the multi-agent trading system.
    
    Attributes:
        market: Market environment instance
        app: Dash application instance
        port: Port to run the dashboard on
    """
    
    def __init__(self, market, port=8050):
        """
        Initialize the dashboard.
        
        Args:
            market: Market environment instance
            port: Port to run the dashboard on
        """
        self.market = market
        self.port = port
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        
        # Set up the layout
        self._setup_layout()
        
        # Set up callbacks
        self._setup_callbacks()
        self.app = dash.Dash(__name__) 
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
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
            
            # Store components for data
            dcc.Store(id="market-data-store"),
            dcc.Store(id="agent-data-store"),
            dcc.Interval(id="simulation-interval", interval=1000, n_intervals=0, disabled=True)
        ], fluid=True, className="mt-4")
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            Output("simulation-interval", "disabled"),
            Output("simulation-status", "children"),
            Input("start-button", "n_clicks"),
            Input("pause-button", "n_clicks"),
            Input("reset-button", "n_clicks"),
            State("simulation-interval", "disabled"),
            prevent_initial_call=True
        )
        def control_simulation(start_clicks, pause_clicks, reset_clicks, is_disabled):
            ctx = dash.callback_context
            if not ctx.triggered:
                return is_disabled, "Simulation not started"
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            if button_id == "start-button":
                return False, "Simulation running"
            elif button_id == "pause-button":
                return True, "Simulation paused"
            elif button_id == "reset-button":
                self.market.reset()
                return True, "Simulation reset"
            
            return is_disabled, "Simulation status unknown"
        
        @self.app.callback(
            Output("simulation-interval", "interval"),
            Input("simulation-speed", "value")
        )
        def update_interval(speed):
            # Convert speed (1-10) to interval in ms (2000-200)
            return 2200 - (speed * 200)
        
        @self.app.callback(
            Output("market-data-store", "data"),
            Output("agent-data-store", "data"),
            Input("simulation-interval", "n_intervals")
        )
        def update_simulation(n_intervals):
            # Run one step of the simulation
            step_result = self.market.step()
            
            # Get agent performance
            agent_performance = self.market.get_agent_performance()
            
            # Return updated data
            return step_result, agent_performance
        
        @self.app.callback(
            Output("market-graph", "figure"),
            Input("market-data-store", "data")
        )
        def update_market_graph(market_data):
            if not market_data:
                # Return empty figure if no data
                return go.Figure()
            
            # Create figure with price data
            fig = go.Figure()
            
            # Get historical window for display
            historical_data = self.market.get_historical_window(30)
            
            # Check if historical_data is empty or doesn't have the required column
            if historical_data.empty or 'symbol' not in historical_data.columns:
                fig.update_layout(
                    title="No market data available",
                    template="plotly_dark"
                )
                return fig
            
            # Group by symbol and date
            for symbol in self.market.symbols:
                symbol_data = historical_data[historical_data['symbol'] == symbol]
                
                if not symbol_data.empty:
                    fig.add_trace(go.Scatter(
                        x=symbol_data['date'],
                        y=symbol_data['close'],
                        mode='lines',
                        name=symbol
                    ))
            
            fig.update_layout(
                title="Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        
        @self.app.callback(
            Output("performance-graph", "figure"),
            Input("agent-data-store", "data")
        )
        def update_performance_graph(agent_data):
            if not agent_data:
                # Return empty figure if no data
                return go.Figure()
            
            # Create figure with agent performance data
            fig = go.Figure()
            
            # Add bar for each agent's return
            agents = list(agent_data.keys())
            returns = [agent_data[agent]['total_return'] * 100 for agent in agents]
            
            fig.add_trace(go.Bar(
                x=agents,
                y=returns,
                text=[f"{ret:.2f}%" for ret in returns],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Agent Performance (Return %)",
                xaxis_title="Agent",
                yaxis_title="Return (%)",
                template="plotly_dark"
            )
            
            return fig
        
        @self.app.callback(
            Output("agent-actions-graph", "figure"),
            Input("market-data-store", "data")
        )
        def update_actions_graph(market_data):
            if not market_data or 'agent_actions' not in market_data:
                # Return empty figure if no data
                return go.Figure()
            
            # Create figure with agent actions
            fig = go.Figure()
            
            # Get agent actions from market history
            if self.market.market_history:
                # Get the last 10 market states
                recent_history = self.market.market_history[-10:]
                
                # Count buy/sell actions for each agent
                agent_ids = [agent.agent_id for agent in self.market.agents]
                buy_counts = {agent_id: 0 for agent_id in agent_ids}
                sell_counts = {agent_id: 0 for agent_id in agent_ids}
                
                for state in recent_history:
                    for agent_id, actions in state.get('agent_actions', {}).items():
                        for symbol, action_data in actions.items():
                            if action_data['action'] == 'buy':
                                buy_counts[agent_id] += 1
                            elif action_data['action'] == 'sell':
                                sell_counts[agent_id] += 1
                
                # Create grouped bar chart
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
                    title="Recent Agent Actions",
                    xaxis_title="Agent",
                    yaxis_title="Count",
                    barmode='group',
                    template="plotly_dark",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
            
            return fig
        
        @self.app.callback(
            Output("portfolio-graph", "figure"),
            Input("agent-data-store", "data")
        )
        def update_portfolio_graph(agent_data):
            if not agent_data:
                # Return empty figure if no data
                return go.Figure()
            
            # Create figure with portfolio composition
            fig = go.Figure()
            
            # For demonstration, show a simple portfolio breakdown for the first agent
            if self.market.agents:
                agent = self.market.agents[0]
                portfolio = agent.portfolio
                
                if portfolio:
                    labels = list(portfolio.keys())
                    values = [portfolio[symbol] * self.market.current_prices.get(symbol, 0) for symbol in labels]
                    
                    # Add cash as well
                    labels.append('Cash')
                    values.append(agent.cash)
                    
                    fig.add_trace(go.Pie(
                        labels=labels,
                        values=values,
                        hole=.3
                    ))
                    
                    fig.update_layout(
                        title=f"Portfolio Composition - {agent.name}",
                        template="plotly_dark"
                    )
                else:
                    # Only cash
                    fig.add_trace(go.Pie(
                        labels=['Cash'],
                        values=[agent.cash],
                        hole=.3
                    ))
                    
                    fig.update_layout(
                        title=f"Portfolio Composition - {agent.name} (Cash Only)",
                        template="plotly_dark"
                    )
            
            return fig
        
        @self.app.callback(
            Output("agent-details-content", "children"),
            Input("agent-tabs", "active_tab"),
            Input("agent-data-store", "data")
        )
        def update_agent_details(active_tab, agent_data):
            if not agent_data:
                return html.P("No agent data available")
            
            # Find the agent based on the active tab
            agent_type = active_tab.split("-")[1]  # Extract agent type from tab ID
            
            for agent in self.market.agents:
                if agent_type in agent.agent_id:
                    # Create a details panel for the selected agent
                    metrics = agent.performance_metrics
                    
                    return html.Div([
                        html.H4(agent.name),
                        html.P(f"Agent ID: {agent.agent_id}"),
                        html.P(f"Cash: ${agent.cash:.2f}"),
                        html.P(f"Portfolio Value: ${sum([agent.portfolio.get(symbol, 0) * self.market.current_prices.get(symbol, 0) for symbol in agent.portfolio]):.2f}"),
                        html.P(f"Total Return: {metrics['total_return'] * 100:.2f}%"),
                        html.P(f"Trades Count: {metrics['trades_count']}"),
                        
                        html.H5("Recent Transactions", className="mt-3"),
                        html.Div([
                            dbc.Table.from_dataframe(
                                pd.DataFrame(agent.transaction_history[-5:]) if agent.transaction_history else pd.DataFrame(),
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm"
                            )
                        ])
                    ])
            
            return html.P(f"No {agent_type} agent found")
    
    def run(self, debug=False):
        """
        Run the dashboard.
        
        Args:
            debug: Whether to run in debug mode
        """
        # Fixed: Changed run_server to run to match newer Dash API
        self.app.run(debug=debug, port=self.port)
