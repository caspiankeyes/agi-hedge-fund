"""
PortfolioManager - Recursive Meta-Agent Arbitration Framework

This module implements the portfolio meta-agent that recursively arbitrates between
different philosophical investment agents and manages the overall portfolio allocation.

Key capabilities:
- Multi-agent arbitration with philosophical weighting
- Attribution-weighted position sizing
- Recursive consensus formation across agents
- Transparent decision tracing with interpretability scaffolding
- Conflict resolution through value attribution
- Memory-based temporal reasoning across market cycles

Internal Note: The portfolio manager implements the meta-agent arbitration layer
using recursive attribution traces and symbolic consensus formation shells.
"""

import datetime
import uuid
import logging
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import numpy as np
from collections import defaultdict

# Core agent functionality
from ..agents.base import BaseAgent, AgentSignal
from ..cognition.graph import ReasoningGraph
from ..cognition.memory import MemoryShell
from ..cognition.attribution import AttributionTracer
from ..utils.diagnostics import TracingTools

# Type hints
from pydantic import BaseModel, Field


class Position(BaseModel):
    """Current portfolio position with attribution."""
    
    ticker: str = Field(...)
    quantity: int = Field(...)
    entry_price: float = Field(...)
    current_price: float = Field(...)
    entry_date: datetime.datetime = Field(default_factory=datetime.datetime.now)
    attribution: Dict[str, float] = Field(default_factory=dict)  # Agent contributions
    confidence: float = Field(default=0.5)
    reasoning: str = Field(default="")
    value_basis: str = Field(default="")
    last_update: datetime.datetime = Field(default_factory=datetime.datetime.now)


class Portfolio(BaseModel):
    """Portfolio state with positions and performance metrics."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    positions: Dict[str, Position] = Field(default_factory=dict)
    cash: float = Field(...)
    initial_capital: float = Field(...)
    last_update: datetime.datetime = Field(default_factory=datetime.datetime.now)
    performance_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def get_value(self, price_data: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash."""
        total_value = self.cash
        
        for ticker, position in self.positions.items():
            # Get current price if available, otherwise use stored price
            current_price = price_data.get(ticker, position.current_price)
            position_value = position.quantity * current_price
            total_value += position_value
        
        return total_value
    
    def get_returns(self) -> Dict[str, float]:
        """Calculate portfolio returns."""
        if not self.performance_history:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            }
        
        # Extract portfolio values
        values = [entry["portfolio_value"] for entry in self.performance_history]
        
        # Calculate returns
        if len(values) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
            }
        
        # Calculate total return
        total_return = (values[-1] / values[0]) - 1
        
        # Calculate daily returns
        daily_returns = []
        for i in range(1, len(values)):
            daily_return = (values[i] / values[i-1]) - 1
            daily_returns.append(daily_return)
        
        # Calculate annualized return (assuming daily values)
        days = len(values) - 1
        annualized_return = ((1 + total_return) ** (365 / days)) - 1
        
        # Calculate volatility (annualized standard deviation of returns)
        if daily_returns:
            daily_volatility = np.std(daily_returns)
            annualized_volatility = daily_volatility * (252 ** 0.5)  # Assuming 252 trading days
        else:
            annualized_volatility = 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
        }
    
    def get_allocation(self) -> Dict[str, float]:
        """Get current portfolio allocation percentages."""
        total_value = self.cash
        for ticker, position in self.positions.items():
            total_value += position.quantity * position.current_price
        
        if total_value <= 0:
            return {"cash": 1.0}
        
        # Calculate allocations
        allocations = {"cash": self.cash / total_value}
        
        for ticker, position in self.positions.items():
            position_value = position.quantity * position.current_price
            allocations[ticker] = position_value / total_value
        
        return allocations
    
    def update_prices(self, price_data: Dict[str, float]) -> None:
        """Update position prices with latest market data."""
        for ticker, position in self.positions.items():
            if ticker in price_data:
                position.current_price = price_data[ticker]
                position.last_update = datetime.datetime.now()
        
        self.last_update = datetime.datetime.now()
    
    def record_performance(self, price_data: Dict[str, float]) -> Dict[str, Any]:
        """Record current performance snapshot."""
        # Calculate portfolio value
        portfolio_value = self.get_value(price_data)
        
        # Calculate returns
        returns = {
            "daily_return": 0.0,
            "total_return": (portfolio_value / self.initial_capital) - 1,
        }
        
        # Calculate daily return if we have past data
        if self.performance_history:
            last_value = self.performance_history[-1]["portfolio_value"]
            returns["daily_return"] = (portfolio_value / last_value) - 1
        
        # Create snapshot
        snapshot = {
            "timestamp": datetime.datetime.now(),
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "positions": {ticker: pos.dict() for ticker, pos in self.positions.items()},
            "returns": returns,
            "allocation": self.get_allocation(),
        }
        
        # Add to history
        self.performance_history.append(snapshot)
        
        return snapshot


class PortfolioManager:
    """
    Portfolio Meta-Agent for investment arbitration and management.
    
    The PortfolioManager serves as a recursive meta-agent that:
    - Arbitrates between different philosophical agents
    - Forms consensus through attribution-weighted aggregation
    - Manages portfolio allocation and position sizing
    - Provides transparent decision tracing
    - Maintains temporal memory across market cycles
    """
    
    def __init__(
        self,
        agents: List[BaseAgent],
        initial_capital: float = 100000.0,
        arbitration_depth: int = 2,
        max_position_size: float = 0.2,  # 20% max allocation to single position
        min_position_size: float = 0.01,  # 1% min allocation to single position
        consensus_threshold: float = 0.6,  # Minimum confidence for consensus
        show_trace: bool = False,
        risk_budget: float = 0.5,  # Risk budget (0-1)
    ):
        """
        Initialize portfolio manager.
        
        Args:
            agents: List of investment agents
            initial_capital: Starting capital amount
            arbitration_depth: Depth of arbitration reasoning
            max_position_size: Maximum position size as fraction of portfolio
            min_position_size: Minimum position size as fraction of portfolio
            consensus_threshold: Minimum confidence for consensus
            show_trace: Whether to show reasoning traces
            risk_budget: Risk budget (0-1)
        """
        self.id = str(uuid.uuid4())
        self.agents = agents
        self.arbitration_depth = arbitration_depth
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.consensus_threshold = consensus_threshold
        self.show_trace = show_trace
        self.risk_budget = risk_budget
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            cash=initial_capital,
            initial_capital=initial_capital,
        )
        
        # Initialize cognitive components
        self.memory_shell = MemoryShell(decay_rate=0.1)  # Slower decay for meta-agent
        self.attribution_tracer = AttributionTracer()
        
        # Initialize reasoning graph
        self.reasoning_graph = ReasoningGraph(
            agent_name="PortfolioMetaAgent",
            agent_philosophy="Recursive arbitration across philosophical perspectives",
            model_router=agents[0].llm if agents else None,  # Use first agent's model router
            trace_enabled=show_trace,
        )
        
        # Configure meta-agent reasoning graph
        self._configure_reasoning_graph()
        
        # Diagnostics
        self.tracer = TracingTools(agent_id=self.id, agent_name="PortfolioMetaAgent")
        
        # Agent weight tracking
        self.agent_weights = {agent.id: 1.0 / len(agents) for agent in agents} if agents else {}
        
        # Initialize meta-agent state
        self.meta_state = {
            "agent_consensus": {},
            "agent_performance": {},
            "conflict_history": [],
            "arbitration_history": [],
            "risk_budget_used": 0.0,
            "last_rebalance": datetime.datetime.now(),
            "consistency_metrics": {},
        }
        
        # Internal symbolic processing commands
        self._commands = {
            "reflect.trace": self._reflect_trace,
            "fork.signal": self._fork_signal,
            "collapse.detect": self._collapse_detect,
            "attribute.weight": self._attribute_weight,
            "drift.observe": self._drift_observe,
        }
    
    def _configure_reasoning_graph(self) -> None:
        """Configure the meta-agent reasoning graph."""
        # Configure nodes for meta-agent reasoning
        self.reasoning_graph.add_node(
            "generate_agent_signals",
            self._generate_agent_signals
        )
        
        self.reasoning_graph.add_node(
            "consensus_formation",
            self._consensus_formation
        )
        
        self.reasoning_graph.add_node(
            "conflict_resolution",
            self._conflict_resolution
        )
        
        self.reasoning_graph.add_node(
            "position_sizing",
            self._position_sizing
        )
        
        self.reasoning_graph.add_node(
            "meta_reflection",
            self._meta_reflection
        )
        
        # Configure graph structure
        self.reasoning_graph.set_entry_point("generate_agent_signals")
        self.reasoning_graph.add_edge("generate_agent_signals", "consensus_formation")
        self.reasoning_graph.add_edge("consensus_formation", "conflict_resolution")
        self.reasoning_graph.add_edge("conflict_resolution", "position_sizing")
        self.reasoning_graph.add_edge("position_sizing", "meta_reflection")
    
    def process_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data through all agents and form meta-agent consensus.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Processed market data with meta-agent insights
        """
        # Update portfolio prices
        if "tickers" in market_data:
            price_data = {ticker: data.get("price", 0) 
                        for ticker, data in market_data.get("tickers", {}).items()}
            self.portfolio.update_prices(price_data)
        
        # Process market data through each agent
        agent_analyses = {}
        for agent in self.agents:
            try:
                agent_analysis = agent.process_market_data(market_data)
                agent_analyses[agent.id] = {
                    "agent": agent.name,
                    "analysis": agent_analysis,
                    "philosophy": agent.philosophy,
                }
            except Exception as e:
                logging.error(f"Error processing market data with agent {agent.name}: {e}")
        
        # Generate agent signals
        agent_signals = {}
        for agent in self.agents:
            try:
                agent_processed_data = agent_analyses.get(agent.id, {}).get("analysis", {})
                signals = agent.generate_signals(agent_processed_data)
                agent_signals[agent.id] = {
                    "agent": agent.name,
                    "signals": signals,
                    "confidence": np.mean([s.confidence for s in signals]) if signals else 0.5,
                }
            except Exception as e:
                logging.error(f"Error generating signals with agent {agent.name}: {e}")
        
        # Prepare reasoning input
        reasoning_input = {
            "market_data": market_data,
            "agent_analyses": agent_analyses,
            "agent_signals": agent_signals,
            "portfolio": self.portfolio.dict(),
            "agent_weights": self.agent_weights,
            "meta_state": self.meta_state,
        }
        
        # Run meta-agent reasoning
        meta_result = self.reasoning_graph.run(
            input=reasoning_input,
            trace_depth=self.arbitration_depth
        )
        
        # Extract consensus decisions
        consensus_decisions = meta_result.get("output", {}).get("consensus_decisions", [])
        
        # Add to memory
        self.memory_shell.add_experience({
            "type": "market_analysis",
            "market_data": market_data,
            "meta_result": meta_result,
            "timestamp": datetime.datetime.now().isoformat(),
        })
        
        # Create processed data result
        processed_data = {
            "timestamp": datetime.datetime.now(),
            "meta_agent": {
                "consensus_decisions": consensus_decisions,
                "confidence": meta_result.get("confidence", 0.5),
                "agent_weights": self.agent_weights.copy(),
            },
            "agents": {agent.name: agent_analyses.get(agent.id, {}).get("analysis", {}) 
                     for agent in self.agents},
            "portfolio_value": self.portfolio.get_value(price_data),
            "allocation": self.portfolio.get_allocation(),
        }
        
        # Add trace if enabled
        if self.show_trace and "trace" in meta_result:
            processed_data["trace"] = meta_result["trace"]
        
        return processed_data
    
    def execute_trades(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute trade decisions and update portfolio.
        
        Args:
            decisions: List of trade decisions
            
        Returns:
            Trade execution results
        """
        execution_results = {
            "trades": [],
            "errors": [],
            "portfolio_update": {},
            "timestamp": datetime.datetime.now(),
        }
        
        # Get current prices (use stored prices if not available)
        price_data = {ticker: position.current_price 
                    for ticker, position in self.portfolio.positions.items()}
        
        # Execute each decision
        for decision in decisions:
            ticker = decision.get("ticker", "")
            action = decision.get("action", "")
            quantity = decision.get("quantity", 0)
            confidence = decision.get("confidence", 0.5)
            reasoning = decision.get("reasoning", "")
            attribution = decision.get("attribution", {})
            value_basis = decision.get("value_basis", "")
            
            # Skip invalid decisions
            if not ticker or not action or quantity <= 0:
                execution_results["errors"].append({
                    "ticker": ticker,
                    "error": "Invalid decision parameters",
                    "decision": decision,
                })
                continue
            
            # Get current price
            current_price = price_data.get(ticker, 0)
            
            # Fetch from market if not available
            if current_price <= 0:
                # In a real implementation, this would fetch from market
                # For now, use placeholder
                current_price = 100.0
                price_data[ticker] = current_price
            
            try:
                if action == "buy":
                    # Check if we have enough cash
                    cost = quantity * current_price
                    if cost > self.portfolio.cash:
                        max_quantity = math.floor(self.portfolio.cash / current_price)
                        if max_quantity <= 0:
                            execution_results["errors"].append({
                                "ticker": ticker,
                                "error": "Insufficient cash for purchase",
                                "attempted_quantity": quantity,
                                "available_cash": self.portfolio.cash,
                            })
                            continue
                        
                        # Adjust quantity
                        quantity = max_quantity
                        cost = quantity * current_price
                    
                    # Execute buy
                    if ticker in self.portfolio.positions:
                        # Update existing position
                        position = self.portfolio.positions[ticker]
                        new_quantity = position.quantity + quantity
                        new_cost = (position.quantity * position.entry_price) + cost
                        
                        # Calculate new average entry price
                        new_entry_price = new_cost / new_quantity if new_quantity > 0 else current_price
                        
                        # Update position
                        position.quantity = new_quantity
                        position.entry_price = new_entry_price
                        position.current_price = current_price
                        position.last_update = datetime.datetime.now()
                        
                        # Update attribution (weighted by quantity)
                        old_weight = position.quantity / new_quantity
                        new_weight = quantity / new_quantity
                        
                        for agent_id, weight in attribution.items():
                            position.attribution[agent_id] = (
                                (position.attribution.get(agent_id, 0) * old_weight) +
                                (weight * new_weight)
                            )
                        
                        # Update other fields
                        position.confidence = (position.confidence * old_weight) + (confidence * new_weight)
                        position.reasoning += f"\nAdditional purchase: {reasoning}"
                        position.value_basis = value_basis if value_basis else position.value_basis
                    else:
                        # Create new position
                        self.portfolio.positions[ticker] = Position(
                            ticker=ticker,
                            quantity=quantity,
                            entry_price=current_price,
                            current_price=current_price,
                            attribution=attribution,
                            confidence=confidence,
                            reasoning=reasoning,
                            value_basis=value_basis,
                        )
                    
                    # Update cash
                    self.portfolio.cash -= cost
                    
                    # Record trade
                    execution_results["trades"].append({
                        "ticker": ticker,
                        "action": "buy",
                        "quantity": quantity,
                        "price": current_price,
                        "cost": cost,
                        "timestamp": datetime.datetime.now(),
                    })
                
                elif action == "sell":
                    # Check if we have the position
                    if ticker not in self.portfolio.positions:
                        execution_results["errors"].append({
                            "ticker": ticker,
                            "error": "Position not found",
                            "attempted_action": "sell",
                        })
                        continue
                    
                    position = self.portfolio.positions[ticker]
                    
                    # Check if we have enough shares
                    if quantity > position.quantity:
                        quantity = position.quantity
                    
                    # Calculate proceeds
                    proceeds = quantity * current_price
                    
                    # Execute sell
                    if quantity == position.quantity:
                        # Sell entire position
                        del self.portfolio.positions[ticker]
                    else:
                        # Partial sell
                        position.quantity -= quantity
                        position.last_update = datetime.datetime.now()
                    
                    # Update cash
                    self.portfolio.cash += proceeds
                    
                    # Record trade
                    execution_results["trades"].append({
                        "ticker": ticker,
                        "action": "sell",
                        "quantity": quantity,
                        "price": current_price,
                        "proceeds": proceeds,
                        "timestamp": datetime.datetime.now(),
                    })
            
            except Exception as e:
                execution_results["errors"].append({
                    "ticker": ticker,
                    "error": str(e),
                    "decision": decision,
                })
        
        # Update portfolio timestamps
        self.portfolio.last_update = datetime.datetime.now()
        
        # Record performance
        performance_snapshot = self.portfolio.record_performance(price_data)
        execution_results["portfolio_update"] = performance_snapshot
        
        # Update agent states based on trades
        self._update_agent_states(execution_results)
        
        return execution_results
    
    def _update_agent_states(self, execution_results: Dict[str, Any]) -> None:
        """
        Update agent states based on trade results.
        
        Args:
            execution_results: Trade execution results
        """
        # Create feedback for each agent
        for agent in self.agents:
            # Extract agent-specific trades
            agent_trades = []
            for trade in execution_results.get("trades", []):
                ticker = trade.get("ticker", "")
                
                if ticker in self.portfolio.positions:
                    position = self.portfolio.positions[ticker]
                    agent_attribution = position.attribution.get(agent.id, 0)
                    
                    if agent_attribution > 0:
                        agent_trades.append({
                            **trade,
                            "attribution": agent_attribution,
                        })
            
            # Create market feedback
            market_feedback = {
                "trades": agent_trades,
                "portfolio_value": execution_results.get("portfolio_update", {}).get("portfolio_value", 0),
                "timestamp": datetime.datetime.now(),
            }
            
            # Add performance metrics if available
            if "performance" in execution_results.get("portfolio_update", {}):
                market_feedback["performance"] = execution_results["portfolio_update"]["performance"]
            
            # Update agent state
            try:
                agent.update_state(market_feedback)
            except Exception as e:
                logging.error(f"Error updating state for agent {agent.name}: {e}")
    
    def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> Dict[str, Any]:
        """
        Rebalance portfolio to match target allocation.
        
        Args:
            target_allocation: Target allocation as fraction of portfolio
            
        Returns:
            Rebalance results
        """
        rebalance_results = {
            "trades": [],
            "errors": [],
            "initial_allocation": self.portfolio.get_allocation(),
            "target_allocation": target_allocation,
            "timestamp": datetime.datetime.now(),
        }
        
        # Validate target allocation
        total_allocation = sum(target_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow small rounding errors
            rebalance_results["errors"].append({
                "error": "Invalid target allocation, must sum to 1.0",
                "total": total_allocation,
            })
            return rebalance_results
        
        # Get current portfolio value and allocation
        current_value = self.portfolio.get_value({
            ticker: pos.current_price for ticker, pos in self.portfolio.positions.items()
        })
        current_allocation = self.portfolio.get_allocation()
        
        # Calculate trades needed
        trade_decisions = []
        
        # Process sells first (to free up cash)
        for ticker, position in list(self.portfolio.positions.items()):
            current_ticker_allocation = current_allocation.get(ticker, 0)
            target_ticker_allocation = target_allocation.get(ticker, 0)
            
            # Check if we need to sell
            if current_ticker_allocation > target_ticker_allocation:
                # Calculate how much to sell
                current_position_value = position.quantity * position.current_price
                target_position_value = current_value * target_ticker_allocation
                value_to_sell = current_position_value - target_position_value
                
                # Convert to quantity
                quantity_to_sell = math.floor(value_to_sell / position.current_price)
                
                if quantity_to_sell > 0:
                    # Create sell decision
                    trade_decisions.append({
                        "ticker": ticker,
                        "action": "sell",
                        "quantity": min(quantity_to_sell, position.quantity),  # Ensure we don't sell more than we have
                        "confidence": 0.8,  # High confidence for rebalancing
                        "reasoning": f"Portfolio rebalancing to target allocation of {target_ticker_allocation:.1%}",
                        "attribution": position.attribution,  # Maintain attribution
                        "value_basis": "Portfolio efficiency and risk management",
                    })
        
        # Execute sells
        sell_results = self.execute_trades([d for d in trade_decisions if d["action"] == "sell"])
        rebalance_results["trades"].extend(sell_results.get("trades", []))
        rebalance_results["errors"].extend(sell_results.get("errors", []))
        
        # Update cash value after sells
        current_value = self.portfolio.get_value({
            ticker: pos.current_price for ticker, pos in self.portfolio.positions.items()
        })
        
        # Process buys
        buy_decisions = []
        for ticker, target_alloc in target_allocation.items():
            # Skip cash
            if ticker == "cash":
                continue
            
            current_ticker_allocation = 0
            if ticker in self.portfolio.positions:
                position = self.portfolio.positions[ticker]
                current_ticker_allocation = (position.quantity * position.current_price) / current_value
            
            # Check if we need to buy
            if current_ticker_allocation < target_alloc:
                # Calculate how much to buy
                target_position_value = current_value * target_alloc
                current_position_value = 0
                if ticker in self.portfolio.positions:
                    position = self.portfolio.positions[ticker]
                    current_position_value = position.quantity * position.current_price
                
                value_to_buy = target_position_value - current_position_value
                
                # Check if we have enough cash
                if value_to_buy > self.portfolio.cash:
                    value_to_buy = self.portfolio.cash  # Limit to available cash
                
                # Get current price
                current_price = 0
                if ticker in self.portfolio.positions:
                    current_price = self.portfolio.positions[ticker].current_price
                else:
                    # This would fetch from market in a real implementation
                    # For now, use placeholder
                    current_price = 100.0
                
                # Convert to quantity
                quantity_to_buy = math.floor(value_to_buy / current_price)
                
                if quantity_to_buy > 0:
                    # Determine attribution based on existing position or equal weights
                    attribution = {}
                    if ticker in self.portfolio.positions:
                        attribution = self.portfolio.positions[ticker].attribution
                    else:
                        # Equal attribution to all agents
                        for agent in self.agents:
                            attribution[agent.id] = 1.0 / len(self.agents)
                    
                    # Create buy decision
                    buy_decisions.append({
                        "ticker": ticker,
                        "action": "buy",
                        "quantity": quantity_to_buy,
                        "confidence": 0.8,  # High confidence for rebalancing
                        "reasoning": f"Portfolio rebalancing to target allocation of {target_alloc:.1%}",
                        "attribution": attribution,
                        "value_basis": "Portfolio efficiency and risk management",
                    })
        
        # Execute buys
        buy_results = self.execute_trades(buy_decisions)
        rebalance_results["trades"].extend(buy_results.get("trades", []))
        rebalance_results["errors"].extend(buy_results.get("errors", []))
        
        # Record final allocation
        rebalance_results["final_allocation"] = self.portfolio.get_allocation()
        
        # Update last rebalance timestamp
        self.meta_state["last_rebalance"] = datetime.datetime.now()
        
        return rebalance_results
    
    def run_simulation(self, start_date: str, end_date: str, 
                     data_source: str = "yahoo", rebalance_frequency: str = "monthly") -> Dict[str, Any]:
        """
        Run portfolio simulation over a time period.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_source: Market data source
            rebalance_frequency: Rebalance frequency
            
        Returns:
            Simulation results
        """
        # This is a placeholder implementation
        # A real implementation would fetch historical data and simulate day by day
        
        simulation_results = {
            "start_date": start_date,
            "end_date": end_date,
            "data_source": data_source,
            "rebalance_frequency": rebalance_frequency,
            "initial_capital": self.portfolio.initial_capital,
            "final_value": self.portfolio.initial_capital,  # Placeholder
            "trades": [],
            "performance": [],
            "timestamp": datetime.datetime.now(),
        }
        
        # In a real implementation, this would fetch historical data
        # and simulate trading day by day
        
        return simulation_results
    
    def get_portfolio_state(self) -> Dict[str, Any]:
        """
        Get current portfolio state.
        
        Returns:
            Portfolio state
        """
        # Get current prices
        price_data = {ticker: position.current_price 
                    for ticker, position in self.portfolio.positions.items()}
        
        # Calculate portfolio value
        portfolio_value = self.portfolio.get_value(price_data)
        
        # Calculate returns
        returns = self.portfolio.get_returns()
        
        # Calculate allocation
        allocation = self.portfolio.get_allocation()
        
        # Compile portfolio state
        portfolio_state = {
            "portfolio_value": portfolio_value,
            "cash": self.portfolio.cash,
            "positions": {ticker: {
                "ticker": pos.ticker,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "market_value": pos.quantity * pos.current_price,
                "allocation": allocation.get(ticker, 0),
                "unrealized_gain": (pos.current_price / pos.entry_price - 1) * 100,  # Percentage
                "attribution": pos.attribution,
                "entry_date": pos.entry_date.isoformat(),
            } for ticker, pos in self.portfolio.positions.items()},
            "returns": returns,
            "allocation": allocation,
            "initial_capital": self.portfolio.initial_capital,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return portfolio_state
    
    def visualize_consensus_graph(self) -> Dict[str, Any]:
        """
        Generate visualization data for consensus formation graph.
        
        Returns:
            Consensus graph visualization data
        """
        visualization_data = {
            "nodes": [],
            "links": [],
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Add meta-agent node
        visualization_data["nodes"].append({
            "id": "meta",
            "label": "Portfolio Meta-Agent",
            "type": "meta",
            "size": 20,
        })
        
        # Add agent nodes
        for agent in self.agents:
            visualization_data["nodes"].append({
                "id": agent.id,
                "label": f"{agent.name} Agent",
                "type": "agent",
                "philosophy": agent.philosophy,
                "size": 15,
                "weight": self.agent_weights.get(agent.id, 0),
            })
            
            # Add link from agent to meta
            visualization_data["links"].append({
                "source": agent.id,
                "target": "meta",
                "value": self.agent_weights.get(agent.id, 0),
                "type": "influence",
            })
        
        # Add position nodes
        for ticker, position in self.portfolio.positions.items():
            visualization_data["nodes"].append({
                "id": f"position-{ticker}",
                "label": ticker,
                "type": "position",
                "size": 10,
                "value": position.quantity * position.current_price,
            })
            
            # Add link from meta to position
            visualization_data["links"].append({
                "source": "meta",
                "target": f"position-{ticker}",
                "value": 1.0,
                "type": "allocation",
            })
            
            # Add links from agents to position based on attribution
            for agent_id, weight in position.attribution.items():
                if weight > 0.01:  # Threshold to reduce clutter
                    visualization_data["links"].append({
                        "source": agent_id,
                        "target": f"position-{ticker}",
                        "value": weight,
                        "type": "attribution",
                    })
        
        return visualization_data
    
    def visualize_agent_conflict_map(self) -> Dict[str, Any]:
        """
        Generate visualization data for agent conflict map.
        
        Returns:
            Agent conflict map visualization data
        """
        conflict_data = {
            "nodes": [],
            "links": [],
            "conflict_zones": [],
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Add agent nodes
        for agent in self.agents:
            conflict_data["nodes"].append({
                "id": agent.id,
                "label": f"{agent.name} Agent",
                "type": "agent",
                "philosophy": agent.philosophy,
                "size": 15,
            })
        
        # Add position nodes
        for ticker, position in self.portfolio.positions.items():
            conflict_data["nodes"].append({
                "id": f"position-{ticker}",
                "label": ticker,
                "type": "position",
                "size": 10,
            })
        
        # Get recent conflicts from meta state
        conflicts = self.meta_state.get("conflict_history", [])[-10:]
        
        # Add conflict zones
        for conflict in conflicts:
            conflict_data["conflict_zones"].append({
                "id": conflict.get("id", str(uuid.uuid4())),
                "ticker": conflict.get("ticker", ""),
                "agents": conflict.get("agents", []),
                "resolution": conflict.get("resolution", "unresolved"),
                "timestamp": conflict.get("timestamp", datetime.datetime.now().isoformat()),
            })
            
            # Add links between conflicting agents
            agent_ids = conflict.get("agents", [])
            for i in range(len(agent_ids)):
                for j in range(i + 1, len(agent_ids)):
                    conflict_data["links"].append({
                        "source": agent_ids[i],
                        "target": agent_ids[j],
                        "value": 1.0,
                        "type": "conflict",
                        "ticker": conflict.get("ticker", ""),
                    })
        
        return conflict_data
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """
        Calculate performance metrics for each agent.
        
        Returns:
            Agent performance metrics
        """
        agent_performance = {}
        
        # Calculate attribution-weighted returns
        for agent in self.agents:
            # Initialize metrics
            metrics = {
                "total_attribution": 0.0,
                "weighted_return": 0.0,
                "positions": [],
                "win_rate": 0.0,
                "confidence": 0.0,
            }
            
            # Get agent attribution for each position
            position_count = 0
            winning_positions = 0
            
            for ticker, position in self.portfolio.positions.items():
                agent_attribution = position.attribution.get(agent.id, 0)
                
                if agent_attribution > 0:
                    # Calculate position return
                    position_return = (position.current_price / position.entry_price) - 1
                    
                    # Add to metrics
                    metrics["total_attribution"] += agent_attribution
                    metrics["weighted_return"] += position_return * agent_attribution
                    
                    # Track win/loss
                    position_count += 1
                    if position_return > 0:
                        winning_positions += 1
                    
                    # Add position details
                    metrics["positions"].append({
                        "ticker": ticker,
                        "attribution": agent_attribution,
                        "return": position_return,
                        "weight": position.quantity * position.current_price,
                    })
            
            # Calculate win rate
            metrics["win_rate"] = winning_positions / position_count if position_count > 0 else 0
            
            # Get agent confidence
            metrics["confidence"] = agent.state.confidence_history[-1] if agent.state.confidence_history else 0.5
            
            # Calculate weighted return
            if metrics["total_attribution"] > 0:
                metrics["weighted_return"] /= metrics["total_attribution"]
            
            # Store metrics
            agent_performance[agent.id] = {
                "agent": agent.name,
                "philosophy": agent.philosophy,
                "metrics": metrics,
            }
        
        return agent_performance
    
    def save_state(self, filepath: str) -> None:
        """
        Save portfolio manager state to file.
        
        Args:
            filepath: Path to save state
        """
        # Compile state
        state = {
            "id": self.id,
            "portfolio": self.portfolio.dict(),
            "agent_weights": self.agent_weights,
            "meta_state": self.meta_state,
            "arbitration_depth": self.arbitration_depth,
            "max_position_size": self.max_position_size,
            "min_position_size": self.min_position_size,
            "consensus_threshold": self.consensus_threshold,
            "risk_budget": self.risk_budget,
            "memory_shell": self.memory_shell.export_state(),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, filepath: str) -> None:
        """
        Load portfolio manager state from file.
        
        Args:
            filepath: Path to load state from
        """
        # Load from file
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Update state
        self.id = state.get("id", self.id)
        self.agent_weights = state.get("agent_weights", self.agent_weights)
        self.meta_state = state.get("meta_state", self.meta_state)
        self.arbitration_depth = state.get("arbitration_depth", self.arbitration_depth)
        self.max_position_size = state.get("max_position_size", self.max_position_size)
        self.min_position_size = state.get("min_position_size", self.min_position_size)
        self.consensus_threshold = state.get("consensus_threshold", self.consensus_threshold)
        self.risk_budget = state.get("risk_budget", self.risk_budget)
        
        # Load portfolio
        if "portfolio" in state:
            from pydantic import parse_obj_as
            self.portfolio = parse_obj_as(Portfolio, state["portfolio"])
        
        # Load memory shell
        if "memory_shell" in state:
            self.memory_shell.import_state(state["memory_shell"])
    
    # Reasoning graph node implementations
    def _generate_agent_signals(self, state) -> Dict[str, Any]:
        """
        Generate signals from all agents.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Input already contains agent signals
        input_data = state.input
        agent_signals = input_data.get("agent_signals", {})
        
        # Organize signals by ticker
        ticker_signals = defaultdict(list)
        
        for agent_id, agent_data in agent_signals.items():
            for signal in agent_data.get("signals", []):
                ticker = signal.ticker
                ticker_signals[ticker].append({
                    "agent_id": agent_id,
                    "agent_name": agent_data.get("agent", "Unknown"),
                    "signal": signal,
                })
        
        # Return updated context
        return {
            "context": {
                **state.context,
                "ticker_signals": dict(ticker_signals),
                "agent_signals": agent_signals,
            }
        }
    
    def _consensus_formation(self, state) -> Dict[str, Any]:
        """
        Form consensus from agent signals.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Extract signals by ticker
        ticker_signals = state.context.get("ticker_signals", {})
        
        # Form consensus for each ticker
        consensus_decisions = []
        
        for ticker, signals in ticker_signals.items():
            # Skip if no signals
            if not signals:
                continue
            
            # Collect buy/sell/hold signals
            buy_signals = []
            sell_signals = []
            hold_signals = []
            
            for item in signals:
                signal = item.get("signal", {})
                action = signal.action.lower()
                
                if action == "buy":
                    buy_signals.append((item, signal))
                elif action == "sell":
                    sell_signals.append((item, signal))
                elif action == "hold":
                    hold_signals.append((item, signal))
            
            # Skip if conflicting signals (handle in conflict resolution)
            if (buy_signals and sell_signals) or (not buy_signals and not sell_signals and not hold_signals):
                continue
            
            # Form consensus for non-conflicting signals
            if buy_signals:
                # Form buy consensus
                consensus = self._form_action_consensus(ticker, "buy", buy_signals)
                if consensus:
                    consensus_decisions.append(consensus)
            
            elif sell_signals:
                # Form sell consensus
                consensus = self._form_action_consensus(ticker, "sell", sell_signals)
                if consensus:
                    consensus_decisions.append(consensus)
        
        # Return updated output
        return {
            "context": {
                **state.context,
                "consensus_decisions": consensus_decisions,
                "consensus_tickers": [decision.get("ticker") for decision in consensus_decisions],
            },
            "output": {
                "consensus_decisions": consensus_decisions,
            }
        }
    
    def _form_action_consensus(self, ticker: str, action: str, 
                            signals: List[Tuple[Dict[str, Any], Any]]) -> Optional[Dict[str, Any]]:
        """
        Form consensus for a specific action on a ticker.
        
        Args:
            ticker: Stock ticker
            action: Action ("buy" or "sell")
            signals: List of (agent_data, signal) tuples
            
        Returns:
            Consensus decision or None if no consensus
        """
        if not signals:
            return None
        
        # Calculate weighted confidence
        total_weight = 0.0
        weighted_confidence = 0.0
        attribution = {}
        
        for item, signal in signals:
            agent_id = item.get("agent_id", "")
            agent_name = item.get("agent_name", "Unknown")
            
            # Skip if missing agent ID
            if not agent_id:
                continue
            
            # Get agent weight
            agent_weight = self.agent_weights.get(agent_id, 0)
            
            # Add to attribution
            attribution[agent_id] = agent_weight
            
            # Add to weighted confidence
            weighted_confidence += signal.confidence * agent_weight
            total_weight += agent_weight
        
        # Check if we have sufficient weight
        if total_weight <= 0:
            return None
        
        # Normalize attribution
        for agent_id in attribution:
            attribution[agent_id] /= total_weight
        
        # Calculate consensus confidence
        consensus_confidence = weighted_confidence / total_weight
        
        # Check against threshold
        if consensus_confidence < self.consensus_threshold:
            return None
        
        # Aggregate quantities
        total_quantity = sum(signal.quantity for _, signal in signals if hasattr(signal, "quantity") and signal.quantity is not None)
        avg_quantity = total_quantity // len(signals) if signals else 0
        
        # Use majority quantity if significant variation
        quantities = [signal.quantity for _, signal in signals if hasattr(signal, "quantity") and signal.quantity is not None]
        if quantities and max(quantities) / (min(quantities) or 1) > 3:
            # High variation, use median
            quantities.sort()
            median_quantity = quantities[len(quantities) // 2]
        else:
            # Low variation, use average
            median_quantity = avg_quantity
        
        # Combine reasoning
        reasoning_parts = [f"{item.get('agent_name', 'Agent')}: {signal.reasoning}" 
                         for item, signal in signals]
        combined_reasoning = "\n".join(reasoning_parts)
        
        # Get most common value basis (weighted by confidence)
        value_bases = {}
        for item, signal in signals:
            value_basis = signal.value_basis
            weight = signal.confidence * self.agent_weights.get(item.get("agent_id", ""), 0)
            
            if value_basis in value_bases:
                value_bases[value_basis] += weight
            else:
                value_bases[value_basis] = weight
        
        # Get highest weighted value basis
        value_basis = max(value_bases.items(), key=lambda x: x[1])[0] if value_bases else ""
        
        # Create consensus decision
        consensus_decision = {
            "ticker": ticker,
            "action": action,
            "quantity": median_quantity,
            "confidence": consensus_confidence,
            "reasoning": f"Consensus from multiple agents:\n{combined_reasoning}",
            "attribution": attribution,
            "value_basis": value_basis,
        }
        
        return consensus_decision
    
    def _conflict_resolution(self, state) -> Dict[str, Any]:
        """
        Resolve conflicts between agent signals.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Extract ticker signals and consensus decisions
        ticker_signals = state.context.get("ticker_signals", {})
        consensus_decisions = state.context.get("consensus_decisions", [])
        consensus_tickers = state.context.get("consensus_tickers", [])
        
        # Identify tickers with conflicts
        conflict_tickers = []
        
        for ticker, signals in ticker_signals.items():
            # Skip if ticker already has consensus
            if ticker in consensus_tickers:
                continue
            
            # Check for conflicts
            actions = set()
            for item in signals:
                signal = item.get("signal", {})
                actions.add(signal.action.lower())
            
            # Ticker has conflicting actions
            if len(actions) > 1:
                conflict_tickers.append(ticker)
        
        # Resolve each conflict
        resolved_conflicts = []
        
        for ticker in conflict_tickers:
            signals = ticker_signals.get(ticker, [])
            
            # Group signals by action
            action_signals = defaultdict(list)
            
            for item in signals:
                signal = item.get("signal", {})
                action = signal.action.lower()
                action_signals[action].append((item, signal))
            
            # Resolve conflict
            resolution = self._resolve_ticker_conflict(ticker, action_signals)
            
            if resolution:
                # Add to resolved conflicts
                resolved_conflicts.append(resolution)
                
                # Add to consensus decisions
                consensus_decisions.append(resolution)
                
                # Record conflict in meta state
                conflict_record = {
                    "id": str(uuid.uuid4()),
                    "ticker": ticker,
                    "agents": [item.get("agent_id") for item, _ in sum(action_signals.values(), [])],
                    "resolution": "resolved",
                    "action": resolution.get("action"),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                
                self.meta_state["conflict_history"].append(conflict_record)
        
        # Return updated output
        return {
            "context": {
                **state.context,
                "consensus_decisions": consensus_decisions,
                "resolved_conflicts": resolved_conflicts,
            },
            "output": {
                "consensus_decisions": consensus_decisions,
            }
        }
    
    def _resolve_ticker_conflict(self, ticker: str, action_signals: Dict[str, List[Tuple[Dict[str, Any], Any]]]) -> Optional[Dict[str, Any]]:
        """
        Resolve conflict for a specific ticker.
        
        Args:
            ticker: Stock ticker
            action_signals: Dictionary mapping actions to lists of (agent_data, signal) tuples
            
        Returns:
            Resolved decision or None if no resolution
        """
        # Calculate total weight for each action
        action_weights = {}
        action_confidences = {}
        
        for action, signals in action_signals.items():
            total_weight = 0.0
            weighted_confidence = 0.0
            
            for item, signal in signals:
                agent_id = item.get("agent_id", "")
                
                # Skip if missing agent ID
                if not agent_id:
                    continue
                
                # Get agent weight
                agent_weight = self.agent_weights.get(agent_id, 0)
                
                # Add to weighted confidence
                weighted_confidence += signal.confidence * agent_weight
                total_weight += agent_weight
            
            # Store action weight and confidence
            if total_weight > 0:
                action_weights[action] = total_weight
                action_confidences[action] = weighted_confidence / total_weight
        
        # Check if any actions
        if not action_weights:
            return None
        
        # Choose action with highest weight
        best_action = max(action_weights.items(), key=lambda x: x[1])[0]
        
        # Check confidence threshold
        if action_confidences.get(best_action, 0) < self.consensus_threshold:
            return None
        
        # Get signals for best action
        best_signals = action_signals.get(best_action, [])
        
        # Form consensus for best action
        return self._form_action_consensus(ticker, best_action, best_signals)
    
    def _position_sizing(self, state) -> Dict[str, Any]:
        """
        Size positions for consensus decisions.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Extract consensus decisions
        consensus_decisions = state.context.get("consensus_decisions", [])
        
        # Get current portfolio value
        current_portfolio = state.input.get("portfolio", {})
        current_value = current_portfolio.get("cash", 0)
        
        for position in current_portfolio.get("positions", {}).values():
            current_value += position.get("quantity", 0) * position.get("current_price", 0)
        
        # Adjust position sizes
        sized_decisions = []
        
        for decision in consensus_decisions:
            ticker = decision.get("ticker", "")
            action = decision.get("action", "")
            confidence = decision.get("confidence", 0.5)
            
            # Skip if missing ticker or action
            if not ticker or not action:
                continue
            
            # Get current position if exists
            current_position = None
            for position in current_portfolio.get("positions", {}).values():
                if position.get("ticker") == ticker:
                    current_position = position
                    break
            
            # Determine target position size
            target_size = self._calculate_position_size(
                ticker=ticker,
                action=action,
                confidence=confidence,
                attribution=decision.get("attribution", {}),
                portfolio_value=current_value,
            )
            
            # Convert to quantity
            # In a real implementation, this would use current price from market
            current_price = 0
            if current_position:
                current_price = current_position.get("current_price", 0)
            else:
                # This would fetch from market in a real implementation
                # For now, use placeholder
                current_price = 100.0
            
            if current_price <= 0:
                continue
            
            # Convert target size to quantity
            target_quantity = int(target_size / current_price)
            
            # Adjust for existing position
            if current_position and action == "buy":
                # Add to existing position
                current_quantity = current_position.get("quantity", 0)
                target_quantity = max(0, target_quantity - current_quantity)
            
            # Ensure minimum quantity
            if target_quantity <= 0 and action == "buy":
                continue
            
            # Update decision quantity
            decision["quantity"] = target_quantity
            
            # Add to sized decisions
            sized_decisions.append(decision)
        
        # Return updated output
        return {
            "context": {
                **state.context,
                "sized_decisions": sized_decisions,
            },
            "output": {
                "consensus_decisions": sized_decisions,
            }
        }
    
    def _calculate_position_size(self, ticker: str, action: str, confidence: float,
                              attribution: Dict[str, float], portfolio_value: float) -> float:
        """
        Calculate position size based on confidence and attribution.
        
        Args:
            ticker: Stock ticker
            action: Action ("buy" or "sell")
            confidence: Decision confidence
            attribution: Attribution to agents
            portfolio_value: Current portfolio value
            
        Returns:
            Target position size in currency units
        """
        # Base position size as percentage of portfolio
        base_size = self.min_position_size + (confidence * (self.max_position_size - self.min_position_size))
        
        # Adjust for action
        if action == "sell":
            # For sell, use existing position size or default
            for position in self.portfolio.positions.values():
                if position.ticker == ticker:
                    return position.quantity * position.current_price
            
            return 0  # No position to sell
        
        # Calculate attribution-weighted size
        if attribution:
            # Calculate agent performance scores
            performance_scores = {}
            for agent_id, weight in attribution.items():
                # Find agent
                agent = None
                for a in self.agents:
                    if a.id == agent_id:
                        agent = a
                        break
                
                if agent:
                    # Use consistency score as proxy for performance
                    performance_score = agent.state.consistency_score
                    performance_scores[agent_id] = performance_score
            
            # Calculate weighted performance score
            weighted_score = 0
            total_weight = 0
            
            for agent_id, weight in attribution.items():
                if agent_id in performance_scores:
                    weighted_score += performance_scores[agent_id] * weight
                    total_weight += weight
            
            # Adjust base size by performance
            if total_weight > 0:
                performance_factor = weighted_score / total_weight
                base_size *= (0.5 + (0.5 * performance_factor))
        
        # Calculate currency amount
        target_size = portfolio_value * base_size
        
        return target_size
    
    def _meta_reflection(self, state) -> Dict[str, Any]:
        """
        Perform meta-reflection on decision process.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Extract decisions
        sized_decisions = state.context.get("sized_decisions", [])
        
        # Update meta state with arbitration record
        arbitration_record = {
            "id": str(uuid.uuid4()),
            "decisions": sized_decisions,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        self.meta_state["arbitration_history"].append(arbitration_record)
        
        # Update agent weights based on performance
        self._update_agent_weights()
        
        # Calculate meta-confidence
        meta_confidence = sum(decision.get("confidence", 0) for decision in sized_decisions) / len(sized_decisions) if sized_decisions else 0.5
        
        # Return final output
        return {
            "output": {
                "consensus_decisions": sized_decisions,
                "meta_confidence": meta_confidence,
                "agent_weights": self.agent_weights,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "confidence": meta_confidence,
        }
    
    def _update_agent_weights(self) -> None:
        """Update agent weights based on performance."""
        # Get agent performance metrics
        agent_performance = self.get_agent_performance()
        
        # Update agent weights
        for agent_id, performance in agent_performance.items():
            metrics = performance.get("metrics", {})
            
            # Calculate performance score
            weighted_return = metrics.get("weighted_return", 0)
            win_rate = metrics.get("win_rate", 0)
            confidence = metrics.get("confidence", 0.5)
            
            # Combine metrics into single score
            performance_score = (0.5 * weighted_return) + (0.3 * win_rate) + (0.2 * confidence)
            
            # Update meta state
            self.meta_state["agent_performance"][agent_id] = {
                "weighted_return": weighted_return,
                "win_rate": win_rate,
                "confidence": confidence,
                "performance_score": performance_score,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        
        # Calculate new weights
        new_weights = {}
        total_score = 0
        
        for agent_id, performance in self.meta_state["agent_performance"].items():
            score = performance.get("performance_score", 0)
            
            # Ensure non-negative score
            score = max(0.1, score + 0.5)  # Add offset to handle negative returns
            
            new_weights[agent_id] = score
            total_score += score
        
        # Normalize weights
        if total_score > 0:
            for agent_id in new_weights:
                new_weights[agent_id] /= total_score
        
        # Update weights (smooth transition)
        for agent_id, weight in new_weights.items():
            current_weight = self.agent_weights.get(agent_id, 0)
            self.agent_weights[agent_id] = current_weight * 0.7 + weight * 0.3
    
    # Internal command implementations
    def _reflect_trace(self, agent=None, depth=2) -> Dict[str, Any]:
        """
        Trace portfolio meta-agent reflection.
        
        Args:
            agent: Optional agent to reflect on
            depth: Reflection depth
            
        Returns:
            Reflection trace
        """
        if agent:
            # Find agent
            target_agent = None
            for a in self.agents:
                if a.name.lower() == agent.lower() or a.id == agent:
                    target_agent = a
                    break
            
            if target_agent:
                # Delegate to agent's reflect trace
                return target_agent.execute_command("reflect.trace", depth=depth)
        
        # Reflect on meta-agent
        # Get recent arbitration history
        arbitration_history = self.meta_state.get("arbitration_history", [])[-depth:]
        
        # Get agent weights
        agent_weights = self.agent_weights.copy()
        
        # Get conflict history
        conflict_history = self.meta_state.get("conflict_history", [])[-depth:]
        
        # Form reflection
        reflection = {
            "arbitration_history": arbitration_history,
            "agent_weights": agent_weights,
            "conflict_history": conflict_history,
            "meta_agent_description": "Portfolio meta-agent for recursive arbitration across philosophical agents",
            "reflection_depth": depth,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return reflection
    
    def _fork_signal(self, source) -> Dict[str, Any]:
        """
        Fork a signal from specified source.
        
        Args:
            source: Source for signal fork
            
        Returns:
            Fork result
        """
        if source == "agents":
            # Fork from all agents
            all_signals = []
            
            for agent in self.agents:
                # Get agent signals
                try:
                    agent_signals = agent.execute_command("fork.signal", source="beliefs")
                    if agent_signals and "signals" in agent_signals:
                        signals = agent_signals["signals"]
                        
                        # Add agent info
                        for signal in signals:
                            signal["agent"] = agent.name
                            signal["agent_id"] = agent.id
                        
                        all_signals.extend(signals)
                except Exception as e:
                    logging.error(f"Error forking signals from agent {agent.name}: {e}")
            
            return {
                "source": "agents",
                "signals": all_signals,
                "count": len(all_signals),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        
        elif source == "memory":
            # Fork from memory shell
            experiences = self.memory_shell.get_recent_memories(limit=3)
            
            # Extract decisions from experiences
            decisions = []
            
            for exp in experiences:
                if "meta_result" in exp.get("content", {}):
                    meta_result = exp["content"]["meta_result"]
                    if "output" in meta_result and "consensus_decisions" in meta_result["output"]:
                        exp_decisions = meta_result["output"]["consensus_decisions"]
                        decisions.extend(exp_decisions)
            
            return {
                "source": "memory",
                "decisions": decisions,
                "count": len(decisions),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        
        else:
            return {
                "error": "Invalid source",
                "source": source,
                "timestamp": datetime.datetime.now().isoformat(),
            }
    
    def _collapse_detect(self, threshold=0.7, reason=None) -> Dict[str, Any]:
        """
        Detect reasoning collapse in meta-agent.
        
        Args:
            threshold: Collapse detection threshold
            reason: Optional specific reason to check
            
        Returns:
            Collapse detection results
        """
        # Check for different collapse conditions
        collapses = {
            "conflict_threshold": len(self.meta_state.get("conflict_history", [])) > 10,
            "agent_weight_skew": max(self.agent_weights.values()) > 0.8 if self.agent_weights else False,
            "consensus_failure": len(self.meta_state.get("arbitration_history", [])) > 0 and 
                              not self.meta_state.get("arbitration_history", [])[-1].get("decisions", []),
        }
        
        # If specific reason provided, check only that
        if reason and reason in collapses:
            collapse_detected = collapses[reason]
            collapse_reasons = {reason: collapses[reason]} if collapse_detected else {}
        else:
            # Check all collapses
            collapse_detected = any(collapses.values())
            collapse_reasons = {k: v for k, v in collapses.items() if v}
        
        return {
            "collapse_detected": collapse_detected,
            "collapse_reasons": collapse_reasons,
            "threshold": threshold,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    
    def _attribute_weight(self, justification) -> Dict[str, Any]:
        """
        Attribute weight to a justification.
        
        Args:
            justification: Justification text
            
        Returns:
            Attribution weight results
        """
        # Extract key themes
        themes = []
        for agent in self.agents:
            if agent.philosophy.lower() in justification.lower():
                themes.append(agent.philosophy)
        
        # Calculate weight for each agent
        agent_weights = {}
        
        for agent in self.agents:
            # Calculate theme alignment
            theme_alignment = 0
            for theme in themes:
                if theme.lower() in agent.philosophy.lower():
                    theme_alignment += 1
            
            theme_alignment = theme_alignment / len(themes) if themes else 0
            
            # Get baseline weight
            baseline_weight = self.agent_weights.get(agent.id, 0)
            
            # Calculate final weight
            if theme_alignment > 0:
                agent_weights[agent.id] = baseline_weight * (1 + theme_alignment)
            else:
                agent_weights[agent.id] = baseline_weight * 0.5
        
        # Normalize weights
        total_weight = sum(agent_weights.values())
        if total_weight > 0:
            for agent_id in agent_weights:
                agent_weights[agent_id] /= total_weight
        
        return {
            "attribution": agent_weights,
            "themes": themes,
            "justification": justification,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    
    def _drift_observe(self, vector, bias=0.0) -> Dict[str, Any]:
        """
        Observe agent drift patterns.
        
        Args:
            vector: Drift vector
            bias: Bias adjustment
            
        Returns:
            Drift observation results
        """
        # Record in meta state
        self.meta_state["agent_drift"] = {
            "vector": vector,
            "bias": bias,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Calculate drift magnitude
        drift_magnitude = sum(abs(v) for v in vector.values()) / len(vector) if vector else 0
        
        # Apply bias
        drift_magnitude += bias
        
        # Check if drift exceeds threshold
        drift_significant = drift_magnitude > 0.3
        
        return {
            "drift_vector": vector,
            "drift_magnitude": drift_magnitude,
            "drift_significant": drift_significant,
            "bias_applied": bias,
            "timestamp": datetime.datetime.now().isoformat(),
        }
    
    def execute_command(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute internal command.
        
        Args:
            command: Command string
            **kwargs: Command parameters
            
        Returns:
            Command execution results
        """
        if command in self._commands:
            return self._commands[command](**kwargs)
        else:
            return {
                "error": "Unknown command",
                "command": command,
                "available_commands": list(self._commands.keys()),
                "timestamp": datetime.datetime.now().isoformat(),
            }
    
    def __repr__(self) -> str:
        """String representation of portfolio manager."""
        return f"PortfolioManager(agents={len(self.agents)}, depth={self.arbitration_depth})"
