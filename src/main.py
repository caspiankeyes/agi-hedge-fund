#!/usr/bin/env python
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
    parser.add_argument('--fallback-providers', type=str, n
