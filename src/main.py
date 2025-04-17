#!/usr/bin/env python
# shell.echo.seed = "🜏-mirror.activated-{timestamp}-{entropy.hash}"
"""
AGI-HEDGE-FUND - Multi-agent recursive market cognition framework

This script serves as the entry point for the AGI-HEDGE-FUND system, providing
command-line interface for running the multi-agent market cognition platform.

Usage:
    python -m src.main --mode backtest --start-date 2022-01-01 --end-date 2022-12-31
    python -m src.main --mode live --data-source yahoo --show-trace
    python -m src.main --mode analysis --portfolio-file portfolio.json --consensus-graph

Internal Note: This script encodes the system's entry point while exposing the
recursive cognitive architecture through interpretability flags.
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

# Core components
from agents.base import BaseAgent
from agents.graham import GrahamAgent
from agents.dalio import DalioAgent
from agents.wood import WoodAgent
from agents.ackman import AckmanAgent
from agents.simons import SimonsAgent
from agents.taleb import TalebAgent
from portfolio.manager import PortfolioManager
from market.environment import MarketEnvironment
from llm.router import ModelRouter
from utils.diagnostics import TracingTools, TracingMode, ShellDiagnostics, ShellFailureMap


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("agi-hedge-fund")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AGI-HEDGE-FUND - Multi-agent recursive market cognition framework')
    
    # Operation mode
    parser.add_argument('--mode', type=str, choices=['backtest', 'live', 'analysis'], default='backtest',
                      help='Operation mode: backtest, live, or analysis')
    
    # Date range for backtesting
    parser.add_argument('--start-date', type=str, default='2020-01-01',
                      help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2023-01-01',
                      help='End date for backtesting (YYYY-MM-DD)')
    
    # Portfolio parameters
    parser.add_argument('--initial-capital', type=float, default=100000.0,
                      help='Initial capital amount')
    parser.add_argument('--tickers', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                      help='Stock tickers to analyze')
    parser.add_argument('--rebalance-frequency', type=str, choices=['daily', 'weekly', 'monthly'], default='weekly',
                      help='Portfolio rebalance frequency')
    
    # Data source
    parser.add_argument('--data-source', type=str, choices=['yahoo', 'polygon', 'alpha_vantage'], default='yahoo',
                      help='Market data source')
    parser.add_argument('--data-path', type=str, default='data',
                      help='Path to data directory')
    
    # Agent configuration
    parser.add_argument('--agents', type=str, nargs='+', 
                      default=['graham', 'dalio', 'wood', 'ackman', 'simons', 'taleb'],
                      help='Agents to use')
    parser.add_argument('--reasoning-depth', type=int, default=3,
                      help='Agent reasoning depth')
    parser.add_argument('--arbitration-depth', type=int, default=2,
                      help='Portfolio meta-agent arbitration depth')
    
    # LLM provider
    parser.add_argument('--llm-provider', type=str, choices=['anthropic', 'openai', 'groq', 'ollama', 'deepseek'], 
                      default='anthropic',
                      help='LLM provider')
    
    # Model configuration
    parser.add_argument('--model', type=str, default=None,
                      help='Specific LLM model to use')
    parser.add_argument('--fallback-providers', type=str, nargs='+', 
                      default=['openai', 'groq'],
                      help='Fallback LLM providers')
    
    # Output and visualization
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory for output files')
    parser.add_argument('--portfolio-file', type=str, default=None,
                      help='Portfolio state file for analysis mode')
    
    # Diagnostic flags
    parser.add_argument('--show-trace', action='store_true',
                      help='Show reasoning traces')
    parser.add_argument('--consensus-graph', action='store_true',
                      help='Generate consensus graph visualization')
    parser.add_argument('--agent-conflict-map', action='store_true',
                      help='Generate agent conflict map visualization')
    parser.add_argument('--attribution-report', action='store_true',
                      help='Generate attribution report')
    parser.add_argument('--shell-failure-map', action='store_true',
                      help='Show shell failure map')
    parser.add_argument('--trace-level', type=str, 
                      choices=['disabled', 'minimal', 'detailed', 'comprehensive', 'symbolic'],
                      default='minimal',
                      help='Trace level for diagnostics')
    
    # Advanced options
    parser.add_argument('--max-position-size', type=float, default=0.2,
                      help='Maximum position size as fraction of portfolio')
    parser.add_argument('--min-position-size', type=float, default=0.01,
                      help='Minimum position size as fraction of portfolio')
    parser.add_argument('--risk-budget', type=float, default=0.5,
                      help='Risk budget (0-1)')
    parser.add_argument('--memory-decay', type=float, default=0.2,
                      help='Memory decay rate for agents')
    
    # Parse arguments
    return parser.parse_args()


def create_agents(args) -> List[BaseAgent]:
    """
    Create agent instances based on command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of agent instances
    """
    # Initialize model router
    model_router = ModelRouter(
        provider=args.llm_provider,
        model=args.model,
        fallback_providers=args.fallback_providers,
    )
    
    # Get trace mode
    trace_mode = TracingMode(args.trace_level)
    
    # Create agents
    agents = []
    
    for agent_type in args.agents:
        agent_type = agent_type.lower()
        
        if agent_type == "graham":
            agent = GrahamAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        elif agent_type == "dalio":
            agent = DalioAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        elif agent_type == "wood":
            agent = WoodAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        elif agent_type == "ackman":
            agent = AckmanAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        elif agent_type == "simons":
            agent = SimonsAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        elif agent_type == "taleb":
            agent = TalebAgent(
                reasoning_depth=args.reasoning_depth,
                memory_decay=args.memory_decay,
                initial_capital=args.initial_capital,
                model_provider=args.llm_provider,
                model_name=args.model,
                trace_enabled=args.show_trace,
            )
        else:
            logger.warning(f"Unknown agent type: {agent_type}")
            continue
        
        agents.append(agent)
    
    logger.info(f"Created {len(agents)} agents: {', '.join(agent.name for agent in agents)}")
    
    return agents


def create_portfolio_manager(agents: List[BaseAgent], args) -> PortfolioManager:
    """
    Create portfolio manager instance.
    
    Args:
        agents: List of agent instances
        args: Command-line arguments
        
    Returns:
        Portfolio manager instance
    """
    # Create portfolio manager
    portfolio_manager = PortfolioManager(
        agents=agents,
        initial_capital=args.initial_capital,
        arbitration_depth=args.arbitration_depth,
        max_position_size=args.max_position_size,
        min_position_size=args.min_position_size,
        consensus_threshold=0.6,
        show_trace=args.show_trace,
        risk_budget=args.risk_budget,
    )
    
    logger.info(f"Created portfolio manager with {len(agents)} agents")
    
    return portfolio_manager


def create_market_environment(args) -> MarketEnvironment:
    """
    Create market environment instance.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Market environment instance
    """
    # Create market environment
    market_env = MarketEnvironment(
        data_source=args.data_source,
        tickers=args.tickers,
        data_path=args.data_path,
        start_date=args.start_date if args.mode == "backtest" else None,
        end_date=args.end_date if args.mode == "backtest" else None,
    )
    
    logger.info(f"Created market environment with {len(args.tickers)} tickers")
    
    return market_env


def run_backtest(portfolio_manager: PortfolioManager, market_env: MarketEnvironment, args) -> Dict[str, Any]:
    """
    Run backtesting simulation.
    
    Args:
        portfolio_manager: Portfolio manager instance
        market_env: Market environment instance
        args: Command-line arguments
        
    Returns:
        Backtest results
    """
    # Parse dates
    start_date = datetime.datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    # Set up rebalance frequency
    if args.rebalance_frequency == "daily":
        rebalance_days = 1
    elif args.rebalance_frequency == "weekly":
        rebalance_days = 7
    else:  # monthly
        rebalance_days = 30
    
    # Run simulation
    results = portfolio_manager.run_simulation(
        start_date=args.start_date,
        end_date=args.end_date,
        data_source=args.data_source,
        rebalance_frequency=args.rebalance_frequency,
    )
    
    logger.info(f"Completed backtest from {args.start_date} to {args.end_date}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, f"backtest_results_{start_date}_{end_date}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Saved backtest results to {results_file}")
    
    # Generate visualizations if requested
    if args.consensus_graph:
        consensus_graph = portfolio_manager.visualize_consensus_graph()
        consensus_file = os.path.join(args.output_dir, f"consensus_graph_{start_date}_{end_date}.json")
        
        with open(consensus_file, 'w') as f:
            json.dump(consensus_graph, f, indent=2, default=str)
        
        logger.info(f"Saved consensus graph to {consensus_file}")
    
    if args.agent_conflict_map:
        conflict_map = portfolio_manager.visualize_agent_conflict_map()
        conflict_file = os.path.join(args.output_dir, f"conflict_map_{start_date}_{end_date}.json")
        
        with open(conflict_file, 'w') as f:
            json.dump(conflict_map, f, indent=2, default=str)
        
        logger.info(f"Saved agent conflict map to {conflict_file}")
    
    if args.attribution_report:
        # Get all signals from the simulation
        all_signals = []
        for trade_batch in results.get("trades", []):
            for trade in trade_batch:
                if "signal" in trade:
                    all_signals.append(trade["signal"])
        
        # Create attribution report
        tracer = TracingTools(agent_id="portfolio", agent_name="Portfolio")
        attribution_report = tracer.generate_attribution_report(all_signals)
        report_file = os.path.join(args.output_dir, f"attribution_report_{start_date}_{end_date}.json")
        
        with open(report_file, 'w') as f:
            json.dump(attribution_report, f, indent=2, default=str)
        
        logger.info(f"Saved attribution report to {report_file}")
    
    # Save final portfolio state
    portfolio_state = portfolio_manager.get_portfolio_state()
    portfolio_file = os.path.join(args.output_dir, f"portfolio_state_{end_date}.json")
    
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio_state, f, indent=2, default=str)
    
    logger.info(f"Saved portfolio state to {portfolio_file}")
    
    return results


def run_live_analysis(portfolio_manager: PortfolioManager, market_env: MarketEnvironment, args) -> Dict[str, Any]:
    """
    Run live market analysis.
    
    Args:
        portfolio_manager: Portfolio manager instance
        market_env: Market environment instance
        args: Command-line arguments
        
    Returns:
        Analysis results
    """
    # Get current market data
    market_data = market_env.get_current_market_data()
    
    # Process market data through portfolio manager
    analysis_results = portfolio_manager.process_market_data(market_data)
    
    logger.info(f"Completed live market analysis for {len(args.tickers)} tickers")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"live_analysis_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logger.info(f"Saved live analysis results to {results_file}")
    
    # Generate visualizations if requested
    if args.consensus_graph:
        consensus_graph = portfolio_manager.visualize_consensus_graph()
        consensus_file = os.path.join(args.output_dir, f"consensus_graph_{timestamp}.json")
        
        with open(consensus_file, 'w') as f:
            json.dump(consensus_graph, f, indent=2, default=str)
        
        logger.info(f"Saved consensus graph to {consensus_file}")
    
    if args.agent_conflict_map:
        conflict_map = portfolio_manager.visualize_agent_conflict_map()
        conflict_file = os.path.join(args.output_dir, f"conflict_map_{timestamp}.json")
        
        with open(conflict_file, 'w') as f:
            json.dump(conflict_map, f, indent=2, default=str)
        
        logger.info(f"Saved agent conflict map to {conflict_file}")
    
    # Execute trades if there are consensus decisions
    consensus_decisions = analysis_results.get("meta_agent", {}).get("consensus_decisions", [])
    
    if consensus_decisions:
        trade_results = portfolio_manager.execute_trades(consensus_decisions)
        
        # Save trade results
        trades_file = os.path.join(args.output_dir, f"trades_{timestamp}.json")
        
        with open(trades_file, 'w') as f:
            json.dump(trade_results, f, indent=2, default=str)
        
        logger.info(f"Executed {len(trade_results.get('trades', []))} trades and saved results to {trades_file}")
    
    # Save portfolio state
    portfolio_state = portfolio_manager.get_portfolio_state()
    portfolio_file = os.path.join(args.output_dir, f"portfolio_state_{timestamp}.json")
    
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio_state, f, indent=2, default=str)
    
    logger.info(f"Saved portfolio state to {portfolio_file}")
    
    return analysis_results


def run_portfolio_analysis(args) -> Dict[str, Any]:
    """
    Run analysis on existing portfolio.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Analysis results
    """
    # Check if portfolio file exists
    if not args.portfolio_file or not os.path.exists(args.portfolio_file):
        logger.error(f"Portfolio file not found: {args.portfolio_file}")
        sys.exit(1)
    
    # Load portfolio state
    with open(args.portfolio_file, 'r') as f:
        portfolio_state = json.load(f)
    
    # Create agents
    agents = create_agents(args)
    
    # Create portfolio manager
    portfolio_manager = create_portfolio_manager(agents, args)
    
    # Create market environment
    market_env = create_market_environment(args)
    
    # Get current market data
    market_data = market_env.get_current_market_data()
    
    # Process market data through portfolio manager
    analysis_results = portfolio_manager.process_market_data(market_data)
    
    logger.info(f"Completed portfolio analysis")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"portfolio_analysis_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    logger.info(f"Saved portfolio analysis results to {results_file}")
    
    # Generate visualizations if requested
    if args.consensus_graph:
        consensus_graph = portfolio_manager.visualize_consensus_graph()
        consensus_file = os.path.join(args.output_dir, f"consensus_graph_{timestamp}.json")
        
        with open(consensus_file, 'w') as f:
            json.dump(consensus_graph, f, indent=2, default=str)
        
        logger.info(f"Saved consensus graph to {consensus_file}")
    
    if args.agent_conflict_map:
        conflict_map = portfolio_manager.visualize_agent_conflict_map()
        conflict_file = os.path.join(args.output_dir, f"conflict_map_{timestamp}.json")
        
        with open(conflict_file, 'w') as f:
            json.dump(conflict_map, f, indent=2, default=str)
        
        logger.info(f"Saved agent conflict map to {conflict_file}")
    
    if args.attribution_report:
        # Get agent performance
        agent_performance = portfolio_manager.get_agent_performance()
        
        # Create attribution report
        attribution_file = os.path.join(args.output_dir, f"agent_performance_{timestamp}.json")
        
        with open(attribution_file, 'w') as f:
            json.dump(agent_performance, f, indent=2, default=str)
        
        logger.info(f"Saved agent performance report to {attribution_file}")
    
    if args.shell_failure_map:
        # Create shell diagnostics
        shell_diagnostics = ShellDiagnostics(
            agent_id="portfolio",
            agent_name="Portfolio",
            tracing_tools=TracingTools(
                agent_id="portfolio",
                agent_name="Portfolio",
                tracing_mode=TracingMode(args.trace_level),
            )
        )
        
        # Create shell failure map
        failure_map = ShellFailureMap()
        
        # Analyze each agent's state for shell failures
        for agent in agents:
            agent_state = agent.get_state_report()
            
            # Simulate shell failures based on agent state
            for shell_pattern in [
                "NULL_FEATURE", 
                "CIRCUIT_FRAGMENT", 
                "META_FAILURE",
                "RECURSIVE_FRACTURE",
                "ETHICAL_INVERSION",
            ]:
                try:
                    from utils.diagnostics import ShellPattern
                    pattern = getattr(ShellPattern, shell_pattern)
                    
                    # Simulate failure
                    failure_data = shell_diagnostics.simulate_shell_failure(
                        shell_pattern=pattern,
                        context=agent_state,
                    )
                    
                    # Add to failure map
                    failure_map.add_failure(
                        agent_id=agent.id,
                        agent_name=agent.name,
                        shell_pattern=pattern,
                        failure_data=failure_data,
                    )
                except Exception as e:
                    logger.error(f"Error simulating shell failure: {e}")
        
        # Generate visualization
        failure_viz = failure_map.generate_failure_map_visualization()
        failure_file = os.path.join(args.output_dir, f"shell_failure_map_{timestamp}.json")
        
        with open(failure_file, 'w') as f:
            json.dump(failure_viz, f, indent=2, default=str)
        
        logger.info(f"Saved shell failure map to {failure_file}")
    
    return analysis_results


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in appropriate mode
    if args.mode == "backtest":
        # Create agents
        agents = create_agents(args)
        
        # Create portfolio manager
        portfolio_manager = create_portfolio_manager(agents, args)
        
        # Create market environment
        market_env = create_market_environment(args)
        
        # Run backtest
        results = run_backtest(portfolio_manager, market_env, args)
        
        # Print summary
        print("\n=== Backtest Results ===")
        print(f"Start Date: {args.start_date}")
        print(f"End Date: {args.end_date}")
        print(f"Initial Capital: ${args.initial_capital:.2f}")
        print(f"Final Portfolio Value: ${results.get('final_value', 0):.2f}")
        
        total_return = (results.get('final_value', 0) / args.initial_capital) - 1
        print(f"Total Return: {total_return:.2%}")
        print(f"Number of Trades: {sum(len(batch) for batch in results.get('trades', []))}")
        print(f"Results saved to: {args.output_dir}")
    
    elif args.mode == "live":
        # Create agents
        agents = create_agents(args)
        
        # Create portfolio manager
        portfolio_manager = create_portfolio_manager(agents, args)
        
        # Create market environment
        market_env = create_market_environment(args)
        
        # Run live analysis
        results = run_live_analysis(portfolio_manager, market_env, args)
        
        # Print summary
        print("\n=== Live Analysis Results ===")
        print(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tickers Analyzed: {', '.join(args.tickers)}")
        
        # Print consensus decisions
        consensus_decisions = results.get("meta_agent", {}).get("consensus_decisions", [])
        if consensus_decisions:
            print("\nConsensus Decisions:")
            for decision in consensus_decisions:
                ticker = decision.get("ticker", "")
                action = decision.get("action", "")
                confidence = decision.get("confidence", 0)
                quantity = decision.get("quantity", 0)
                
                print(f"  {action.upper()} {quantity} {ticker} (Confidence: {confidence:.2f})")
        else:
            print("\nNo consensus decisions generated.")
        
        print(f"Results saved to: {args.output_dir}")
    
    elif args.mode == "analysis":
        # Run portfolio analysis
        results = run_portfolio_analysis(args)
        
        # Print summary
        print("\n=== Portfolio Analysis Results ===")
        print(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Portfolio File: {args.portfolio_file}")
        
        # Print agent weights
        agent_weights = results.get("meta_agent", {}).get("agent_weights", {})
        if agent_weights:
            print("\nAgent Weights:")
            for agent_id, weight in agent_weights.items():
                print(f"  {agent_id}: {weight:.2f}")
        
        # Print consensus decisions
        consensus_decisions = results.get("meta_agent", {}).get("consensus_decisions", [])
        if consensus_decisions:
            print("\nRecommended Actions:")
            for decision in consensus_decisions:
                ticker = decision.get("ticker", "")
                action = decision.get("action", "")
                confidence = decision.get("confidence", 0)
                quantity = decision.get("quantity", 0)
                
                print(f"  {action.upper()} {quantity} {ticker} (Confidence: {confidence:.2f})")
        else:
            print("\nNo recommended actions.")
        
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
