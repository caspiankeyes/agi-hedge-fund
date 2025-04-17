"""
GrahamAgent - Value Investing Cognitive Agent

This module implements Benjamin Graham's value investing philosophy as a 
recursive cognitive agent with specialized market interpretation capabilities.

Key characteristics:
- Focuses on margin of safety and intrinsic value
- Detects undervalued assets based on fundamentals
- Maintains skepticism toward market sentiment
- Prioritizes long-term value over short-term price movements
- Exhibits patience and discipline with high conviction

Internal Notes: The Graham shell simulates the CIRCUIT-FRAGMENT and NULL-FEATURE
shells for detecting undervalued assets and knowledge boundaries.
"""

import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from .base import BaseAgent, AgentSignal
from ..cognition.graph import ReasoningGraph
from ..utils.diagnostics import TracingTools


class GrahamAgent(BaseAgent):
    """
    Agent embodying Benjamin Graham's value investing philosophy.
    
    Implements specialized cognitive patterns for:
    - Intrinsic value calculation
    - Margin of safety evaluation
    - Fundamental analysis
    - Value trap detection
    - Long-term perspective
    """
    
    def __init__(
        self,
        reasoning_depth: int = 3,
        memory_decay: float = 0.1,  # Lower memory decay for long-term perspective
        initial_capital: float = 100000.0,
        margin_of_safety: float = 0.3,  # Minimum discount to intrinsic value
        model_provider: str = "anthropic",
        model_name: str = "claude-3-sonnet-20240229",
        trace_enabled: bool = False,
    ):
        """
        Initialize Graham value investing agent.
        
        Args:
            reasoning_depth: Depth of recursive reasoning
            memory_decay: Rate of memory deterioration
            initial_capital: Starting capital amount
            margin_of_safety: Minimum discount to intrinsic value requirement
            model_provider: LLM provider
            model_name: Specific model identifier
            trace_enabled: Whether to generate full reasoning traces
        """
        super().__init__(
            name="Graham",
            philosophy="Value investing focused on margin of safety and fundamental analysis",
            reasoning_depth=reasoning_depth,
            memory_decay=memory_decay,
            initial_capital=initial_capital,
            model_provider=model_provider,
            model_name=model_name,
            trace_enabled=trace_enabled,
        )
        
        self.margin_of_safety = margin_of_safety
        
        # Value investing specific state
        self.state.reflective_state.update({
            'value_detection_threshold': 0.7,
            'sentiment_skepticism': 0.8,
            'patience_factor': 0.9,
            'fundamental_weighting': 0.8,
            'technical_weighting': 0.2,
        })
        
        # Customize reasoning graph for value investing
        self._configure_reasoning_graph()
    
    def _configure_reasoning_graph(self) -> None:
        """Configure the reasoning graph with value investing specific nodes."""
        self.reasoning_graph.add_node(
            "intrinsic_value_analysis",
            fn=self._intrinsic_value_analysis
        )
        
        self.reasoning_graph.add_node(
            "margin_of_safety_evaluation", 
            fn=self._margin_of_safety_evaluation
        )
        
        self.reasoning_graph.add_node(
            "fundamental_analysis",
            fn=self._fundamental_analysis
        )
        
        self.reasoning_graph.add_node(
            "value_trap_detection",
            fn=self._value_trap_detection
        )
        
        # Configure value investing reasoning flow
        self.reasoning_graph.set_entry_point("intrinsic_value_analysis")
        self.reasoning_graph.add_edge("intrinsic_value_analysis", "margin_of_safety_evaluation")
        self.reasoning_graph.add_edge("margin_of_safety_evaluation", "fundamental_analysis")
        self.reasoning_graph.add_edge("fundamental_analysis", "value_trap_detection")
    
    def process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process market data through Graham's value investing lens.
        
        Focuses on:
        - Extracting fundamental metrics
        - Calculating intrinsic value estimates
        - Identifying margin of safety opportunities
        - Filtering for value characteristics
        
        Args:
            data: Market data dictionary
            
        Returns:
            Processed market data with value investing insights
        """
        processed_data = {
            'timestamp': datetime.datetime.now(),
            'tickers': {},
            'market_sentiment': data.get('market_sentiment', {}),
            'economic_indicators': data.get('economic_indicators', {}),
            'insights': [],
        }
        
        # Process each ticker
        for ticker, ticker_data in data.get('tickers', {}).items():
            # Extract fundamental metrics
            fundamentals = ticker_data.get('fundamentals', {})
            price = ticker_data.get('price', 0)
            
            # Skip if insufficient fundamental data
            if not fundamentals or price == 0:
                processed_data['tickers'][ticker] = {
                    'price': price,
                    'analysis': 'insufficient_data',
                    'intrinsic_value': None,
                    'margin_of_safety': None,
                    'recommendation': 'hold',
                }
                continue
            
            # Calculate intrinsic value (Graham-style)
            intrinsic_value = self._calculate_intrinsic_value(fundamentals, ticker_data)
            
            # Calculate margin of safety
            margin_of_safety = (intrinsic_value - price) / intrinsic_value if intrinsic_value > 0 else 0
            
            # Determine if it's a value opportunity
            is_value_opportunity = margin_of_safety >= self.margin_of_safety
            
            # Check for value traps
            value_trap_indicators = self._detect_value_trap_indicators(fundamentals, ticker_data)
            
            # Generate value-oriented analysis
            analysis = self._generate_value_analysis(
                ticker=ticker,
                fundamentals=fundamentals,
                price=price,
                intrinsic_value=intrinsic_value,
                margin_of_safety=margin_of_safety,
                value_trap_indicators=value_trap_indicators
            )
            
            # Store processed ticker data
            processed_data['tickers'][ticker] = {
                'price': price,
                'intrinsic_value': intrinsic_value,
                'margin_of_safety': margin_of_safety,
                'is_value_opportunity': is_value_opportunity,
                'value_trap_risk': len(value_trap_indicators) / 5 if value_trap_indicators else 0,
                'value_trap_indicators': value_trap_indicators,
                'analysis': analysis,
                'recommendation': 'buy' if is_value_opportunity and not value_trap_indicators else 'hold',
                'fundamentals': fundamentals,
            }
            
            # Generate insight if it's a strong value opportunity
            if is_value_opportunity and not value_trap_indicators and margin_of_safety > 0.4:
                processed_data['insights'].append({
                    'ticker': ticker,
                    'type': 'strong_value_opportunity',
                    'margin_of_safety': margin_of_safety,
                    'intrinsic_value': intrinsic_value,
                    'current_price': price,
                })
        
        # Run reflective trace if enabled
        if self.trace_enabled:
            processed_data['reflection'] = self.execute_command(
                command="reflect.trace",
                agent=self.name,
                depth=self.reasoning_depth
            )
        
        return processed_data
    
    def _calculate_intrinsic_value(self, fundamentals: Dict[str, Any], ticker_data: Dict[str, Any]) -> float:
        """
        Calculate intrinsic value using Graham's methods.
        
        Args:
            fundamentals: Fundamental metrics dict
            ticker_data: Complete ticker data
            
        Returns:
            Estimated intrinsic value
        """
        # Extract key metrics
        eps = fundamentals.get('eps', 0)
        book_value = fundamentals.get('book_value_per_share', 0)
        growth_rate = fundamentals.get('growth_rate', 0)
        
        # Graham's formula: IV = EPS * (8.5 + 2g) * 4.4 / Y
        # Where g is growth rate and Y is current AAA bond yield
        # We use a simplified approach here
        bond_yield = ticker_data.get('economic_indicators', {}).get('aaa_bond_yield', 0.045)
        bond_factor = 4.4 / max(bond_yield, 0.01)  # Prevent division by zero
        
        # Calculate growth-adjusted PE
        growth_adjusted_pe = 8.5 + (2 * growth_rate)
        
        # Calculate earnings-based value
        earnings_value = eps * growth_adjusted_pe * bond_factor if eps > 0 else 0
        
        # Calculate book value with margin
        book_value_margin = book_value * 1.5  # Graham often looked for stocks below 1.5x book
        
        # Use the lower of the two values for conservatism
        if earnings_value > 0 and book_value_margin > 0:
            intrinsic_value = min(earnings_value, book_value_margin)
        else:
            intrinsic_value = earnings_value if earnings_value > 0 else book_value_margin
        
        return max(intrinsic_value, 0)  # Ensure non-negative value
    
    def _detect_value_trap_indicators(self, fundamentals: Dict[str, Any], ticker_data: Dict[str, Any]) -> List[str]:
        """
        Detect potential value trap indicators.
        
        Args:
            fundamentals: Fundamental metrics dict
            ticker_data: Complete ticker data
            
        Returns:
            List of value trap indicators
        """
        value_trap_indicators = []
        
        # Check for declining earnings
        if fundamentals.get('earnings_growth', 0) < -0.1:
            value_trap_indicators.append('declining_earnings')
        
        # Check for high debt
        if fundamentals.get('debt_to_equity', 0) > 1.5:
            value_trap_indicators.append('high_debt')
        
        # Check for deteriorating financials
        if fundamentals.get('return_on_equity', 0) < 0.05:
            value_trap_indicators.append('low_return_on_equity')
        
        # Check for industry decline
        if ticker_data.get('sector', {}).get('decline', False):
            value_trap_indicators.append('industry_decline')
        
        # Check for negative free cash flow
        if fundamentals.get('free_cash_flow', 0) < 0:
            value_trap_indicators.append('negative_cash_flow')
        
        return value_trap_indicators
    
    def _generate_value_analysis(self, ticker: str, fundamentals: Dict[str, Any], 
                                price: float, intrinsic_value: float, 
                                margin_of_safety: float, value_trap_indicators: List[str]) -> str:
        """
        Generate value investing analysis summary.
        
        Args:
            ticker: Stock ticker
            fundamentals: Fundamental metrics
            price: Current price
            intrinsic_value: Calculated intrinsic value
            margin_of_safety: Current margin of safety
            value_trap_indicators: List of value trap indicators
            
        Returns:
            Analysis summary text
        """
        # Format for better readability
        iv_formatted = f"${intrinsic_value:.2f}"
        price_formatted = f"${price:.2f}"
        mos_percentage = f"{margin_of_safety * 100:.1f}%"
        
        # Base analysis
        if margin_of_safety >= self.margin_of_safety:
            base_analysis = (f"{ticker} appears undervalued. Current price {price_formatted} vs. "
                           f"intrinsic value estimate {iv_formatted}, providing a "
                           f"{mos_percentage} margin of safety.")
        elif margin_of_safety > 0:
            base_analysis = (f"{ticker} is moderately priced. Current price {price_formatted} vs. "
                           f"intrinsic value estimate {iv_formatted}, providing only a "
                           f"{mos_percentage} margin of safety.")
        else:
            base_analysis = (f"{ticker} appears overvalued. Current price {price_formatted} vs. "
                           f"intrinsic value estimate {iv_formatted}, providing no "
                           f"margin of safety.")
        
        # Add value trap indicators if present
        if value_trap_indicators:
            trap_text = ", ".join(value_trap_indicators)
            base_analysis += f" Warning: Potential value trap indicators detected: {trap_text}."
        
        # Add fundamental highlights
        fundamental_highlights = []
        if fundamentals.get('pe_ratio', 0) > 0:
            fundamental_highlights.append(f"P/E ratio: {fundamentals.get('pe_ratio', 0):.2f}")
        if fundamentals.get('price_to_book', 0) > 0:
            fundamental_highlights.append(f"P/B ratio: {fundamentals.get('price_to_book', 0):.2f}")
        if fundamentals.get('dividend_yield', 0) > 0:
            fundamental_highlights.append(f"Dividend yield: {fundamentals.get('dividend_yield', 0) * 100:.2f}%")
        
        if fundamental_highlights:
            base_analysis += " Key metrics: " + ", ".join(fundamental_highlights) + "."
        
        return base_analysis
    
    def generate_signals(self, processed_data: Dict[str, Any]) -> List[AgentSignal]:
        """
        Generate investment signals based on processed value investing analysis.
        
        Args:
            processed_data: Processed market data with value analysis
            
        Returns:
            List of investment signals with attribution
        """
        signals = []
        
        for ticker, ticker_data in processed_data.get('tickers', {}).items():
            # Skip if insufficient data
            if ticker_data.get('analysis') == 'insufficient_data':
                continue
            
            # Determine action based on value characteristics
            is_value_opportunity = ticker_data.get('is_value_opportunity', False)
            value_trap_risk = ticker_data.get('value_trap_risk', 0)
            margin_of_safety = ticker_data.get('margin_of_safety', 0)
            
            # Skip if no clear signal
            if not is_value_opportunity and margin_of_safety <= 0:
                continue
            
            # Determine action
            if is_value_opportunity and value_trap_risk < 0.3:
                action = 'buy'
                # Scale confidence based on margin of safety
                confidence = min(0.5 + (margin_of_safety * 0.5), 0.95)
                
                # Scale quantity based on conviction
                max_allocation = 0.1  # Max 10% of portfolio in one position
                allocation = max_allocation * confidence
                quantity = int((self.current_capital * allocation) / ticker_data.get('price', 1))
                
                # Ensure minimum quantity
                quantity = max(quantity, 1)
                
                # Create signal dictionary
                signal = {
                    'ticker': ticker,
                    'action': action,
                    'confidence': confidence,
                    'quantity': quantity,
                    'reasoning': f"Value investment opportunity with {margin_of_safety:.1%} margin of safety. {ticker_data.get('analysis', '')}",
                    'intent': "Capitalize on identified value opportunity with sufficient margin of safety",
                    'value_basis': "Intrinsic value significantly exceeds current market price, presenting favorable risk-reward",
                }
                
                signals.append(signal)
            elif margin_of_safety > 0 and margin_of_safety < self.margin_of_safety and value_trap_risk < 0.2:
                # Watchlist signal - lower confidence
                action = 'buy'
                confidence = 0.3 + (margin_of_safety * 0.3)  # Lower confidence
                
                # Smaller position size for watchlist items
                max_allocation = 0.05  # Max 5% of portfolio 
                allocation = max_allocation * confidence
                quantity = int((self.current_capital * allocation) / ticker_data.get('price', 1))
                
                # Ensure minimum quantity
                quantity = max(quantity, 1)
                
                # Create signal dictionary
                signal = {
                    'ticker': ticker,
                    'action': action,
                    'confidence': confidence,
                    'quantity': quantity,
                    'reasoning': f"Moderate value opportunity with {margin_of_safety:.1%} margin of safety. {ticker_data.get('analysis', '')}",
                    'intent': "Establish small position in moderately valued company with potential",
                    'value_basis': "Price below intrinsic value but insufficient margin of safety for full position",
                }
                
                signals.append(signal)
        
        # Apply attribution to signals
        attributed_signals = self.attribute_signals(signals)
        
        # Log trace if enabled
        if self.trace_enabled:
            for signal in attributed_signals:
                self.tracer.record_signal(signal)
        
        return attributed_signals
    
    # Value investing specific reasoning nodes
    def _intrinsic_value_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze intrinsic value of securities.
        
        Args:
            data: Market data
            
        Returns:
            Intrinsic value analysis results
        """
        results = {
            'ticker_valuations': {},
            'timestamp': datetime.datetime.now(),
        }
        
        for ticker, ticker_data in data.get('tickers', {}).items():
            fundamentals = ticker_data.get('fundamentals', {})
            price = ticker_data.get('price', 0)
            
            if not fundamentals or price == 0:
                results['ticker_valuations'][ticker] = {
                    'intrinsic_value': None,
                    'status': 'insufficient_data',
                }
                continue
            
            # Calculate intrinsic value
            intrinsic_value = self._calculate_intrinsic_value(fundamentals, ticker_data)
            
            # Determine valuation status
            if intrinsic_value > price * 1.3:  # 30% above price
                status = 'significantly_undervalued'
            elif intrinsic_value > price * 1.1:  # 10% above price
                status = 'moderately_undervalued'
            elif intrinsic_value > price:
                status = 'slightly_undervalued'
            elif intrinsic_value > price * 0.9:  # Within 10% below price
                status = 'fairly_valued'
            else:
                status = 'overvalued'
            
            results['ticker_valuations'][ticker] = {
                'intrinsic_value': intrinsic_value,
                'price': price,
                'ratio': intrinsic_value / price if price > 0 else 0,
                'status': status,
            }
        
        return results
    
    def _margin_of_safety_evaluation(self, intrinsic_value_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate margin of safety for each security.
        
        Args:
            intrinsic_value_results: Intrinsic value analysis results
            
        Returns:
            Margin of safety evaluation results
        """
        results = {
            'margin_of_safety_analysis': {},
            'value_opportunities': [],
            'timestamp': datetime.datetime.now(),
        }
        
        for ticker, valuation in intrinsic_value_results.get('ticker_valuations', {}).items():
            if valuation.get('status') == 'insufficient_data':
                continue
            
            intrinsic_value = valuation.get('intrinsic_value', 0)
            price = valuation.get('price', 0)
            
            if intrinsic_value <= 0 or price <= 0:
                continue
            
            # Calculate margin of safety
            margin_of_safety = (intrinsic_value - price) / intrinsic_value
            
            # Determine confidence based on margin of safety
            if margin_of_safety >= self.margin_of_safety:
                confidence = min(0.5 + (margin_of_safety * 0.5), 0.95)
                meets_criteria = True
            else:
                confidence = max(0.2, margin_of_safety * 2)
                meets_criteria = False
            
            results['margin_of_safety_analysis'][ticker] = {
                'margin_of_safety': margin_of_safety,
                'meets_criteria': meets_criteria,
                'confidence': confidence,
            }
            
            # Add to value opportunities if meets criteria
            if meets_criteria:
                results['value_opportunities'].append({
                    'ticker': ticker,
                    'margin_of_safety': margin_of_safety,
                    'confidence': confidence,
                    'intrinsic_value': intrinsic_value,
                    'price': price,
                })
        
        # Sort value opportunities by margin of safety
        results['value_opportunities'] = sorted(
            results['value_opportunities'],
            key=lambda x: x['margin_of_safety'],
            reverse=True
        )
        
        return results
    
    def _fundamental_analysis(self, safety_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform fundamental analysis on value opportunities.
        
        Args:
            safety_results: Margin of safety evaluation results
            
        Returns:
            Fundamental analysis results
        """
        results = {
            'fundamental_quality': {},
            'quality_ranking': [],
            'timestamp': datetime.datetime.now(),
        }
        
        # Process each value opportunity
        for opportunity in safety_results.get('value_opportunities', []):
            ticker = opportunity.get('ticker')
            
            # Get ticker data from state
            ticker_data = self.state.working_memory.get('market_data', {}).get('tickers', {}).get(ticker, {})
            fundamentals = ticker_data.get('fundamentals', {})
            
            if not fundamentals:
                continue
            
            # Calculate fundamental quality score
            quality_score = self._calculate_fundamental_quality(fundamentals)
            
            # Store fundamental quality
            results['fundamental_quality'][ticker] = {
                'quality_score': quality_score,
                'roe': fundamentals.get('return_on_equity', 0),
                'debt_to_equity': fundamentals.get('debt_to_equity', 0),
                'current_ratio': fundamentals.get('current_ratio', 0),
                'free_cash_flow': fundamentals.get('free_cash_flow', 0),
                'dividend_history': fundamentals.get('dividend_history', []),
            }
            
            # Add to quality ranking
            results['quality_ranking'].append({
                'ticker': ticker,
                'quality_score': quality_score,
                'margin_of_safety': opportunity.get('margin_of_safety', 0),
                # Combined score weights both quality and value
                'combined_score': quality_score * 0.4 + opportunity.get('margin_of_safety', 0) * 0.6,
            })
        
        # Sort quality ranking by combined score
        results['quality_ranking'] = sorted(
            results['quality_ranking'],
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return results
    
    def _calculate_fundamental_quality(self, fundamentals: Dict[str, Any]) -> float:
        """
        Calculate fundamental quality score.
        
        Args:
            fundamentals: Fundamental metrics
            
        Returns:
            Quality score (0-1)
        """
        # Initialize score
        score = 0.5  # Start at neutral
        
        # Factor 1: Return on Equity (higher is better)
        roe = fundamentals.get('return_on_equity', 0)
        if roe > 0.2:  # Excellent ROE
            score += 0.1
        elif roe > 0.15:  # Very good ROE
            score += 0.075
        elif roe > 0.1:  # Good ROE
            score += 0.05
        elif roe < 0.05:  # Poor ROE
            score -= 0.05
        elif roe < 0:  # Negative ROE
            score -= 0.1
        
        # Factor 2: Debt to Equity (lower is better)
        debt_to_equity = fundamentals.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:  # Very low debt
            score += 0.1
        elif debt_to_equity < 0.5:  # Low debt
            score += 0.05
        elif debt_to_equity > 1.0:  # High debt
            score -= 0.05
        elif debt_to_equity > 1.5:  # Very high debt
            score -= 0.1
        
        # Factor 3: Current Ratio (higher is better)
        current_ratio = fundamentals.get('current_ratio', 0)
        if current_ratio > 3:  # Excellent liquidity
            score += 0.075
        elif current_ratio > 2:  # Very good liquidity
            score += 0.05
        elif current_ratio > 1.5:  # Good liquidity
            score += 0.025
        elif current_ratio < 1:  # Poor liquidity
            score -= 0.1
        
        # Factor 4: Free Cash Flow (positive is better)
        fcf = fundamentals.get('free_cash_flow', 0)
        if fcf > 0:  # Positive FCF
            score += 0.075
        else:  # Negative FCF
            score -= 0.1
        
        # Factor 5: Dividend History (consistent is better)
            dividend_history = fundamentals.get('dividend_history', [])
            if len(dividend_history) >= 5 and all(d > 0 for d in dividend_history):
                # Consistent dividends for 5+ years
                score += 0.075
            elif len(dividend_history) >= 3 and all(d > 0 for d in dividend_history):
                # Consistent dividends for 3+ years
                score += 0.05
            
        # Ensure score is between 0 and 1
        return max(0, min(1, score))
    
    def _value_trap_detection(self, fundamental_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect potential value traps among value opportunities.
        
        Args:
            fundamental_results: Fundamental analysis results
            
        Returns:
            Value trap detection results
        """
        results = {
            'value_trap_analysis': {},
            'safe_opportunities': [],
            'timestamp': datetime.datetime.now(),
        }
        
        # Process each quality-ranked opportunity
        for opportunity in fundamental_results.get('quality_ranking', []):
            ticker = opportunity.get('ticker')
            
            # Get ticker data from state
            ticker_data = self.state.working_memory.get('market_data', {}).get('tickers', {}).get(ticker, {})
            fundamentals = ticker_data.get('fundamentals', {})
            
            if not fundamentals:
                continue
            
            # Detect value trap indicators
            value_trap_indicators = self._detect_value_trap_indicators(fundamentals, ticker_data)
            value_trap_risk = len(value_trap_indicators) / 5 if value_trap_indicators else 0
            
            # Store value trap analysis
            results['value_trap_analysis'][ticker] = {
                'value_trap_risk': value_trap_risk,
                'value_trap_indicators': value_trap_indicators,
                'quality_score': opportunity.get('quality_score', 0),
                'margin_of_safety': opportunity.get('margin_of_safety', 0),
                'combined_score': opportunity.get('combined_score', 0),
            }
            
            # Add to safe opportunities if low value trap risk
            if value_trap_risk < 0.3:
                results['safe_opportunities'].append({
                    'ticker': ticker,
                    'value_trap_risk': value_trap_risk,
                    'quality_score': opportunity.get('quality_score', 0),
                    'margin_of_safety': opportunity.get('margin_of_safety', 0),
                    'combined_score': opportunity.get('combined_score', 0),
                })
        
        # Sort safe opportunities by combined score
        results['safe_opportunities'] = sorted(
            results['safe_opportunities'],
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return results

    def run_analysis_shell(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete analysis shell for Graham value criteria.
        
        This method implements a full CIRCUIT-FRAGMENT shell for detecting
        undervalued assets and NULL-FEATURE shell for knowledge boundaries.
        
        Args:
            market_data: Raw market data
            
        Returns:
            Complete value analysis results
        """
        # Store market data in working memory for value trap detection
        self.state.working_memory['market_data'] = market_data
        
        # Process market data (external interface)
        processed_data = self.process_market_data(market_data)
        
        # Run reasoning graph (internal pipeline)
        initial_results = {'tickers': processed_data.get('tickers', {})}
        
        intrinsic_value_results = self._intrinsic_value_analysis(initial_results)
        safety_results = self._margin_of_safety_evaluation(intrinsic_value_results)
        fundamental_results = self._fundamental_analysis(safety_results)
        trap_results = self._value_trap_detection(fundamental_results)
        
        # Compile complete results
        complete_results = {
            'processed_data': processed_data,
            'intrinsic_value_results': intrinsic_value_results,
            'safety_results': safety_results,
            'fundamental_results': fundamental_results,
            'trap_results': trap_results,
            'final_recommendations': trap_results.get('safe_opportunities', []),
            'timestamp': datetime.datetime.now(),
        }
        
        # Check for collapse conditions
        collapse_check = self.execute_command(
            command="collapse.detect",
            threshold=0.7,
            reason="consistency"
        )
        
        if collapse_check.get('collapse_detected', False):
            complete_results['warnings'] = {
                'collapse_detected': True,
                'collapse_reasons': collapse_check.get('collapse_reasons', {}),
                'message': "Potential inconsistency detected in value analysis process.",
            }
        
        return complete_results
    
    def adjust_strategy(self, performance_metrics: Dict[str, Any]) -> None:
        """
        Adjust Graham strategy based on performance feedback.
        
        Args:
            performance_metrics: Dictionary with performance metrics
        """
        # Extract relevant metrics
        win_rate = performance_metrics.get('win_rate', 0.5)
        avg_return = performance_metrics.get('avg_return', 0)
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        
        # Adjust margin of safety based on win rate
        if win_rate < 0.4:  # Poor win rate
            self.margin_of_safety = min(self.margin_of_safety + 0.05, 0.5)  # Increase safety margin
        elif win_rate > 0.7:  # Excellent win rate
            self.margin_of_safety = max(self.margin_of_safety - 0.05, 0.2)  # Can be less conservative
        
        # Adjust value detection threshold based on returns
        if avg_return < -0.05:  # Significant negative returns
            self.state.reflective_state['value_detection_threshold'] = min(
                self.state.reflective_state.get('value_detection_threshold', 0.7) + 0.05, 
                0.9
            )
        elif avg_return > 0.1:  # Strong positive returns
            self.state.reflective_state['value_detection_threshold'] = max(
                self.state.reflective_state.get('value_detection_threshold', 0.7) - 0.05,
                0.6
            )
        
        # Adjust sentiment skepticism based on drawdown
        if max_drawdown > 0.15:  # Large drawdown
            self.state.reflective_state['sentiment_skepticism'] = min(
                self.state.reflective_state.get('sentiment_skepticism', 0.8) + 0.05,
                0.95
            )
        
        # Update drift vector
        drift_vector = {
            'margin_of_safety': self.margin_of_safety - 0.3,  # Drift from initial value
            'value_detection': self.state.reflective_state.get('value_detection_threshold', 0.7) - 0.7,
            'sentiment_skepticism': self.state.reflective_state.get('sentiment_skepticism', 0.8) - 0.8,
        }
        
        # Observe drift for interpretability
        self.execute_command(
            command="drift.observe",
            vector=drift_vector,
            bias=0.0
        )
    
    def __repr__(self) -> str:
        """Generate string representation of Graham agent."""
        return f"Graham Value Agent (MoS: {self.margin_of_safety:.2f}, Depth: {self.reasoning_depth})"
      
