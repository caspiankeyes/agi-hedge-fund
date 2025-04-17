"""
ReasoningGraph - Recursive Cognitive Architecture

This module implements the core reasoning graph architecture that powers
the recursive cognition capabilities of AGI-HEDGE-FUND agents.

Key components:
- LangGraph-based reasoning networks
- Recursive thought paths with configurable depth
- Attribution tracing throughout reasoning chains
- Symbolic state propagation across reasoning steps
- Failsafe mechanisms for reasoning collapse

Internal Note: This implementation is inspired by circuit interpretability research,
simulating recursive attention pathways via LangGraph. The core patterns
encode .p/ command equivalents like reflect.trace and collapse.detect.
"""

import datetime
import uuid
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Set
import numpy as np

# LangGraph for reasoning graphs
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, ToolExecutor

# For type hints
from pydantic import BaseModel, Field

# Internal imports
from ..llm.router import ModelRouter
from ..utils.diagnostics import TracingTools


class ReasoningState(BaseModel):
    """Reasoning state carried through graph execution."""
    
    # Input data and context
    input: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # Reasoning chain and attribution
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    attribution: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    
    # Execution metadata
    depth: int = Field(default=0)
    max_depth: int = Field(default=3)
    start_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    elapsed_ms: int = Field(default=0)
    
    # Results and conclusions
    output: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.5)
    
    # Error handling and diagnostics
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Collapse detection
    collapse_risk: float = Field(default=0.0)
    collapse_detected: bool = Field(default=False)
    collapse_reason: Optional[str] = Field(default=None)
    
    # Tracing
    trace_enabled: bool = Field(default=False)
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class ReasoningGraph:
    """
    Implements a recursive reasoning architecture for agent cognition.
    
    The ReasoningGraph enables:
    - Multi-step reasoning via connected nodes
    - Recursive thought patterns for deep analysis
    - Attribution tracing across reasoning chains
    - Dynamic graph reconfiguration based on context
    - Safeguards against reasoning collapse (infinite loops, contradictions)
    """
    
    def __init__(
        self,
        agent_name: str,
        agent_philosophy: str,
        model_router: ModelRouter,
        collapse_threshold: float = 0.8,
        trace_enabled: bool = False,
    ):
        """
        Initialize reasoning graph.
        
        Args:
            agent_name: Name of the agent using this reasoning graph
            agent_philosophy: Philosophy description of the agent
            model_router: ModelRouter instance for LLM access
            collapse_threshold: Threshold for reasoning collapse detection
            trace_enabled: Whether to trace reasoning steps
        """
        self.agent_name = agent_name
        self.agent_philosophy = agent_philosophy
        self.model_router = model_router
        self.collapse_threshold = collapse_threshold
        self.trace_enabled = trace_enabled
        
        # Initialize graph components
        self.nodes: Dict[str, Callable] = {}
        self.edges: Dict[str, List[str]] = {}
        self.entry_point: Optional[str] = None
        
        # Diagnostic tooling
        self.tracer = TracingTools(agent_id=str(uuid.uuid4()), agent_name=agent_name)
        
        # Add base nodes
        self._add_base_nodes()
        
        # Build initial graph
        self._build_graph()
    
    def _add_base_nodes(self) -> None:
        """Add base reasoning nodes that are part of every graph."""
        # Entry point for validation
        self.add_node("validate_input", self._validate_input)
        
        # Default nodes
        self.add_node("initial_analysis", self._initial_analysis)
        self.add_node("extract_themes", self._extract_themes)
        self.add_node("generate_conclusions", self._generate_conclusions)
        
        # Exit handlers
        self.add_node("check_collapse", self._check_collapse)
        self.add_node("finalize_output", self._finalize_output)
    
    def _build_graph(self) -> None:
        """Build the reasoning graph based on defined nodes and edges."""
        # Set default entry point if not specified
        if self.entry_point is None:
            if "validate_input" in self.nodes:
                self.entry_point = "validate_input"
            elif len(self.nodes) > 0:
                self.entry_point = list(self.nodes.keys())[0]
            else:
                raise ValueError("Cannot build graph: no nodes defined")
        
        # Build default edges if none defined
        if not self.edges:
            # Get sorted node names (for deterministic graphs)
            node_names = sorted(self.nodes.keys())
            
            # Find entry point index
            try:
                entry_index = node_names.index(self.entry_point)
            except ValueError:
                entry_index = 0
                
            # Move entry point to front
            if entry_index > 0:
                node_names.insert(0, node_names.pop(entry_index))
            
            # Create sequential edges between all nodes
            for i in range(len(node_names) - 1):
                self.add_edge(node_names[i], node_names[i + 1])
            
            # Add exit handler edges
            if "check_collapse" in self.nodes and "check_collapse" not in node_names:
                self.add_edge(node_names[-1], "check_collapse")
                if "finalize_output" in self.nodes:
                    self.add_edge("check_collapse", "finalize_output")
            elif "finalize_output" in self.nodes and "finalize_output" not in node_names:
                self.add_edge(node_names[-1], "finalize_output")
    
    def add_node(self, name: str, fn: Callable) -> None:
        """
        Add a reasoning node to the graph.
        
        Args:
            name: Node name
            fn: Function to execute for this node
        """
        self.nodes[name] = fn
    
    def add_edge(self, source: str, target: str) -> None:
        """
        Add an edge between reasoning nodes.
        
        Args:
            source: Source node name
            target: Target node name
        """
        if source not in self.nodes:
            raise ValueError(f"Source node '{source}' does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node '{target}' does not exist")
        
        if source not in self.edges:
            self.edges[source] = []
        
        if target not in self.edges[source]:
            self.edges[source].append(target)
    
    def set_entry_point(self, node_name: str) -> None:
        """
        Set the entry point for the reasoning graph.
        
        Args:
            node_name: Name of entry node
        """
        if node_name not in self.nodes:
            raise ValueError(f"Entry node '{node_name}' does not exist")
        
        self.entry_point = node_name
    
    def run(self, input: Dict[str, Any], trace_depth: int = 3) -> Dict[str, Any]:
        """
        Run the reasoning graph on input data.
        
        Args:
            input: Input data
            trace_depth: Depth of reasoning trace (higher = deeper reasoning)
            
        Returns:
            Reasoning results
        """
        # Prepare initial state
        state = ReasoningState(
            input=input,
            max_depth=trace_depth,
            trace_enabled=self.trace_enabled
        )
        
        # Define next node function
        def get_next_node(state_dict: Dict[str, Any], current_node: str) -> Union[str, List[str]]:
            # Convert state dict back to ReasoningState
            state = ReasoningState.parse_obj(state_dict)
            
            # Check for collapse detection
            if state.collapse_detected:
                if "finalize_output" in self.nodes:
                    return "finalize_output"
                return END
            
            # Check for max depth reached
            if state.depth >= state.max_depth and current_node != "check_collapse" and "check_collapse" in self.nodes:
                return "check_collapse"
            
            # Check if we've reached a terminal node
            if current_node == "finalize_output" or current_node not in self.edges:
                return END
            
            # Return next nodes
            return self.edges[current_node]
        
        # Create StateGraph
        workflow = StateGraph(ReasoningState)
        
        # Add nodes
        for node_name, node_fn in self.nodes.items():
            workflow.add_node(node_name, self._create_node_wrapper(node_fn))
        
        # Set edge logic
        workflow.set_conditional_edges(
            condition_name="get_next_node",
            condition_fn=get_next_node
        )
        
        # Compile graph
        compiled_graph = workflow.compile()
        
        # Run graph
        start_time = datetime.datetime.now()
        state_dict = compiled_graph.invoke(
            {"input": input, "max_depth": trace_depth, "trace_enabled": self.trace_enabled}
        )
        end_time = datetime.datetime.now()
        
        # Convert state dict back to ReasoningState
        final_state = ReasoningState.parse_obj(state_dict)
        
        # Update execution time
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        final_state.elapsed_ms = execution_time_ms
        
        # Prepare result
        result = {
            "output": final_state.output,
            "confidence": final_state.confidence,
            "elapsed_ms": final_state.elapsed_ms,
            "depth": final_state.depth,
            "collapse_detected": final_state.collapse_detected,
        }
        
        # Include trace if enabled
        if self.trace_enabled:
            result["trace"] = {
                "steps": final_state.steps,
                "attribution": final_state.attribution,
                "errors": final_state.errors,
                "warnings": final_state.warnings,
                "trace_id": final_state.trace_id,
            }
        
        return result
    
    def _create_node_wrapper(self, node_fn: Callable) -> Callable:
        """
        Create a wrapper for node functions to handle tracing and errors.
        
        Args:
            node_fn: Original node function
            
        Returns:
            Wrapped node function
        """
        def wrapped_node(state: Dict[str, Any]) -> Dict[str, Any]:
            # Convert state dict to ReasoningState
            state_obj = ReasoningState.parse_obj(state)
            
            # Increment depth counter
            state_obj.depth += 1
            
            # Run node function with try-except
            try:
                # Get node name from function
                node_name = node_fn.__name__.replace("_", " ").title()
                
                # Record step start
                step_start = {
                    "name": node_name,
                    "depth": state_obj.depth,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                
                # Add to steps
                state_obj.steps.append(step_start)
                
                # Run node function
                result = node_fn(state_obj)
                
                # Update state with result
                if isinstance(result, dict):
                    # Extract and update fields if returned
                    for key, value in result.items():
                        if hasattr(state_obj, key):
                            setattr(state_obj, key, value)
                
                # Record step completion
                step_end = {
                    "name": node_name,
                    "depth": state_obj.depth,
                    "completed": True,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                
                # Update last step
                if state_obj.steps:
                    state_obj.steps[-1].update(step_end)
                
            except Exception as e:
                # Record error
                error = {
                    "message": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "node": node_fn.__name__,
                    "depth": state_obj.depth,
                }
                
                state_obj.errors.append(error)
                
                # Update last step if exists
                if state_obj.steps:
                    state_obj.steps[-1].update({
                        "completed": False,
                        "error": error,
                    })
                
                # Set collapse if critical error
                state_obj.collapse_detected = True
                state_obj.collapse_reason = "critical_error"
            
            # Convert back to dict
            return state_obj.dict()
        
        return wrapped_node
    
    # Base node implementations
    def _validate_input(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Validate input data.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        result = {
            "warnings": state.warnings.copy(),
        }
        
        # Check if input is empty
        if not state.input:
            result["warnings"].append({
                "message": "Empty input provided",
                "severity": "high",
                "timestamp": datetime.datetime.now().isoformat(),
            })
            
            # Set collapse for empty input
            result["collapse_detected"] = True
            result["collapse_reason"] = "empty_input"
        
        return result
    
    def _initial_analysis(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Perform initial analysis of input data.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields
        """
        # Extract key information from input
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "key_entities": [],
            "key_metrics": {},
            "identified_patterns": [],
        }
        
        # Update state
        result = {
            "context": {**state.context, "initial_analysis": analysis},
        }
        
        return result
    
    def _extract_themes(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Extract key themes from the data.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields with extracted themes
        """
        # Extract themes based on agent philosophy
        themes = self.extract_themes(
            text=str(state.input),
            max_themes=5
        )
        
        # Calculate alignment with agent philosophy
        alignment = self.compute_alignment(
            themes=themes,
            philosophy=self.agent_philosophy
        )
        
        # Update result
        result = {
            "context": {
                **state.context,
                "themes": themes,
                "philosophy_alignment": alignment,
            },
        }
        
        return result
    
    def _generate_conclusions(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Generate conclusions based on analysis.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields with conclusions
        """
        # Simple placeholder conclusions
        conclusions = {
            "summary": "Analysis completed",
            "confidence": 0.7,
            "recommendations": [],
        }
        
        # Update result
        result = {
            "output": conclusions,
            "confidence": conclusions["confidence"],
        }
        
        return result
    
    def _check_collapse(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Check for reasoning collapse conditions.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields with collapse detection
        """
        # Default collapse risk is low
        collapse_risk = 0.1
        
        # Check for collapse conditions
        collapse_conditions = {
            "circular_reasoning": self._detect_circular_reasoning(state),
            "confidence_collapse": state.confidence < 0.2,
            "contradiction": self._detect_contradictions(state),
            "depth_exhaustion": state.depth >= state.max_depth,
        }
        
        # Calculate overall collapse risk
        active_conditions = [k for k, v in collapse_conditions.items() if v]
        collapse_risk = len(active_conditions) / len(collapse_conditions) if collapse_conditions else 0
        
        # Determine if collapse detected
        collapse_detected = collapse_risk >= self.collapse_threshold
        collapse_reason = active_conditions[0] if active_conditions else None
        
        # Update result
        result = {
            "collapse_risk": collapse_risk,
            "collapse_detected": collapse_detected,
            "collapse_reason": collapse_reason,
        }
        
        # Add warning if collapse detected
        if collapse_detected:
            result["warnings"] = state.warnings + [{
                "message": f"Reasoning collapse detected: {collapse_reason}",
                "severity": "high",
                "timestamp": datetime.datetime.now().isoformat(),
            }]
        
        return result
    
    def _finalize_output(self, state: ReasoningState) -> Dict[str, Any]:
        """
        Finalize output and prepare result.
        
        Args:
            state: Reasoning state
            
        Returns:
            Updated state fields with finalized output
        """
        # If no output yet, create default
        if not state.output:
            output = {
                "summary": "Analysis completed with limited results",
                "confidence": max(0.1, state.confidence / 2),  # Reduced confidence
                "recommendations": [],
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            # Update result
            result = {
                "output": output,
                "confidence": output["confidence"],
            }
        else:
            # Just return current state unchanged
            result = {}
        
        return result
    
    # Utility methods for reasoning loops
    def extract_themes(self, text: str, max_themes: int = 5) -> List[str]:
        """
        Extract key themes from text using LLM.
        
        Args:
            text: Text to analyze
            max_themes: Maximum number of themes to extract
            
        Returns:
            List of extracted themes
        """
        # Use LLM to extract themes
        prompt = f"""
        Extract the {max_themes} most important themes or topics from the following text.
        Respond with a Python list of strings, each representing one theme.
        
        Text:
        {text}
        
        Themes:
        """
        
        try:
            response = self.model_router.generate(prompt)
            
            # Parse response as Python list (safety handling)
            try:
                themes = eval(response.strip())
                if isinstance(themes, list) and all(isinstance(t, str) for t in themes):
                    return themes[:max_themes]
            except:
                pass
            
            # Fallback parsing
            themes = [line.strip().strip('-*•').strip() 
                     for line in response.split('\n')
                     if line.strip() and line.strip()[0] in '-*•']
            
            return themes[:max_themes]
            
        except Exception as e:
            logging.warning(f"Error extracting themes: {e}")
            return []
    
    def compute_alignment(self, themes: List[str], philosophy: str) -> float:
        """
        Compute alignment between themes and philosophy.
        
        Args:
            themes: List of themes
            philosophy: Philosophy to align with
            
        Returns:
            Alignment score (0-1)
        """
        # Without using LLM, use a simple heuristic based on word overlap
        if not themes:
            return 0.5  # Neutral score for no themes
        
        # Convert to lowercase for comparison
        philosophy_words = set(philosophy.lower().split())
        
        # Count theme words that overlap with philosophy
        alignment_scores = []
        for theme in themes:
            theme_words = set(theme.lower().split())
            overlap = len(theme_words.intersection(philosophy_words))
            total_words = len(theme_words)
            
            # Calculate theme alignment
            if total_words > 0:
                theme_alignment = min(1.0, overlap / (total_words * 0.5))  # Scale for partial matches
                alignment_scores.append(theme_alignment)
        
        # Calculate average alignment across themes
        return sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.5
    
    def run_reflection(self, agent_state: Dict[str, Any], depth: int = 3, 
                      trace_enabled: bool = False) -> Dict[str, Any]:
        """
        Run reflection on agent's current state.
        
        Args:
            agent_state: Agent's current state
            depth: Depth of reflection
            trace_enabled: Whether to enable tracing
            
        Returns:
            Reflection results
        """
        # Prepare reflection input
        reflection_input = {
            "agent_state": agent_state,
            "agent_name": self.agent_name,
            "agent_philosophy": self.agent_philosophy,
            "reflection_depth": depth,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Create reflection prompt for LLM
        prompt = f"""
        You are the {self.agent_name} agent with the following philosophy:
        "{self.agent_philosophy}"
        
        Perform a depth {depth} reflection on your current state and decision making process.
        Focus on:
        1. Consistency between your investment philosophy and current beliefs
        2. Quality of your recent decisions and their alignment with your values
        3. Potential biases or blind spots in your analysis
        4. Areas where your reasoning could be improved
        
        Your current state:
        {agent_state}
        
        Respond with a JSON object containing:
        - assessment: Overall assessment of your cognitive state
        - consistency_score: How consistent your decisions are with your philosophy (0-1)
        - identified_biases: List of potential biases detected
        - improvement_areas: Areas where reasoning could be improved
        - confidence: Your confidence in this self-assessment (0-1)
        """
        
        try:
            # Generate reflection using LLM
            response = self.model_router.generate(prompt)
            
            # Parse JSON response (with fallback)
            # Parse JSON response (with fallback)
            try:
                import json
                reflection = json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing
                reflection = {
                    "assessment": "Unable to parse full reflection",
                    "consistency_score": 0.5,
                    "identified_biases": [],
                    "improvement_areas": [],
                    "confidence": 0.3,
                }
                
                # Extract fields from text response
                if "consistency_score" in response:
                    try:
                        consistency_score = float(response.split("consistency_score")[1].split("\n")[0].replace(":", "").strip())
                        reflection["consistency_score"] = consistency_score
                    except:
                        pass
                
                if "confidence" in response:
                    try:
                        confidence = float(response.split("confidence")[1].split("\n")[0].replace(":", "").strip())
                        reflection["confidence"] = confidence
                    except:
                        pass
            
            return reflection
            
        except Exception as e:
            logging.warning(f"Error in reflection: {e}")
            return {
                "assessment": "Reflection failed due to error",
                "consistency_score": 0.5,
                "identified_biases": ["reflection_failure"],
                "improvement_areas": ["error_handling"],
                "confidence": 0.2,
                "error": str(e),
            }
    
    def generate_from_experiences(self, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate signals based on past experiences.
        
        Args:
            experiences: List of past experiences
            
        Returns:
            List of generated signals
        """
        if not experiences:
            return []
        
        # Format experiences for LLM prompt
        formatted_experiences = "\n".join([
            f"Experience {i+1}: {exp.get('description', 'No description')}" +
            f"\nOutcome: {exp.get('outcome', 'Unknown')}" +
            f"\nTimestamp: {exp.get('timestamp', 'Unknown')}"
            for i, exp in enumerate(experiences)
        ])
        
        # Create generation prompt for LLM
        prompt = f"""
        You are the {self.agent_name} agent with the following philosophy:
        "{self.agent_philosophy}"
        
        Based on the following past experiences, generate potential investment signals:
        
        {formatted_experiences}
        
        Generate 2-3 potential investment signals based on these experiences.
        Each signal should be a JSON object containing:
        - ticker: Stock ticker symbol
        - action: "buy", "sell", or "hold"
        - confidence: Confidence level (0.0-1.0)
        - reasoning: Explicit reasoning chain
        - intent: High-level investment intent
        - value_basis: Core value driving this decision
        
        Respond with a JSON array containing these signals.
        """
        
        try:
            # Generate signals using LLM
            response = self.model_router.generate(prompt)
            
            # Parse JSON response (with fallback)
            try:
                import json
                signals = json.loads(response)
                if not isinstance(signals, list):
                    signals = [signals]
            except json.JSONDecodeError:
                # Simple fallback extraction
                signals = []
                lines = response.split("\n")
                current_signal = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("ticker:") or line.startswith('"ticker":'):
                        # New signal starts
                        if current_signal and "ticker" in current_signal:
                            signals.append(current_signal)
                        current_signal = {}
                        current_signal["ticker"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("action:") or line.startswith('"action":'):
                        current_signal["action"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("confidence:") or line.startswith('"confidence":'):
                        try:
                            current_signal["confidence"] = float(line.split(":", 1)[1].strip().strip('"').strip("'").strip(','))
                        except:
                            current_signal["confidence"] = 0.5
                    elif line.startswith("reasoning:") or line.startswith('"reasoning":'):
                        current_signal["reasoning"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("intent:") or line.startswith('"intent":'):
                        current_signal["intent"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("value_basis:") or line.startswith('"value_basis":'):
                        current_signal["value_basis"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                
                # Add last signal if any
                if current_signal and "ticker" in current_signal:
                    signals.append(current_signal)
            
            # Ensure all required fields are present
            for signal in signals:
                signal.setdefault("ticker", "UNKNOWN")
                signal.setdefault("action", "hold")
                signal.setdefault("confidence", 0.5)
                signal.setdefault("reasoning", "Generated from past experiences")
                signal.setdefault("intent", "Learn from past experiences")
                signal.setdefault("value_basis", "Experiential learning")
            
            return signals
            
        except Exception as e:
            logging.warning(f"Error generating from experiences: {e}")
            return []
    
    def generate_from_beliefs(self, beliefs: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate signals based on current beliefs.
        
        Args:
            beliefs: Current belief state
            
        Returns:
            List of generated signals
        """
        if not beliefs:
            return []
        
        # Format beliefs for LLM prompt
        formatted_beliefs = "\n".join([
            f"Belief: {belief}, Strength: {strength:.2f}"
            for belief, strength in beliefs.items()
        ])
        
        # Create generation prompt for LLM
        prompt = f"""
        You are the {self.agent_name} agent with the following philosophy:
        "{self.agent_philosophy}"
        
        Based on the following current beliefs, generate potential investment signals:
        
        {formatted_beliefs}
        
        Generate 2-3 potential investment signals based on these beliefs.
        Each signal should be a JSON object containing:
        - ticker: Stock ticker symbol (extract from beliefs if present)
        - action: "buy", "sell", or "hold"
        - confidence: Confidence level (0.0-1.0)
        - reasoning: Explicit reasoning chain
        - intent: High-level investment intent
        - value_basis: Core value driving this decision
        
        Respond with a JSON array containing these signals.
        """
        
        try:
            # Generate signals using LLM
            response = self.model_router.generate(prompt)
            
            # Parse JSON response (with fallback)
            try:
                import json
                signals = json.loads(response)
                if not isinstance(signals, list):
                    signals = [signals]
            except json.JSONDecodeError:
                # Simple fallback extraction (same as in generate_from_experiences)
                signals = []
                lines = response.split("\n")
                current_signal = {}
                
                for line in lines:
                    line = line.strip()
                    if line.startswith("ticker:") or line.startswith('"ticker":'):
                        # New signal starts
                        if current_signal and "ticker" in current_signal:
                            signals.append(current_signal)
                        current_signal = {}
                        current_signal["ticker"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("action:") or line.startswith('"action":'):
                        current_signal["action"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("confidence:") or line.startswith('"confidence":'):
                        try:
                            current_signal["confidence"] = float(line.split(":", 1)[1].strip().strip('"').strip("'").strip(','))
                        except:
                            current_signal["confidence"] = 0.5
                    elif line.startswith("reasoning:") or line.startswith('"reasoning":'):
                        current_signal["reasoning"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("intent:") or line.startswith('"intent":'):
                        current_signal["intent"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                    elif line.startswith("value_basis:") or line.startswith('"value_basis":'):
                        current_signal["value_basis"] = line.split(":", 1)[1].strip().strip('"').strip("'").strip(',')
                
                # Add last signal if any
                if current_signal and "ticker" in current_signal:
                    signals.append(current_signal)
            
            # Ensure all required fields are present
            for signal in signals:
                signal.setdefault("ticker", "UNKNOWN")
                signal.setdefault("action", "hold")
                signal.setdefault("confidence", 0.5)
                signal.setdefault("reasoning", "Generated from belief state")
                signal.setdefault("intent", "Act on current beliefs")
                signal.setdefault("value_basis", "Belief consistency")
            
            return signals
            
        except Exception as e:
            logging.warning(f"Error generating from beliefs: {e}")
            return []
    
    # Collapse detection utilities
    def _detect_circular_reasoning(self, state: ReasoningState) -> bool:
        """
        Detect circular reasoning patterns in reasoning steps.
        
        Args:
            state: Current reasoning state
            
        Returns:
            True if circular reasoning detected, False otherwise
        """
        # Need at least 3 steps to detect circularity
        if len(state.steps) < 3:
            return False
        
        # Extract step names
        step_names = [step.get("name", "") for step in state.steps]
        
        # Look for repeating patterns (minimum 2 steps)
        for pattern_len in range(2, len(step_names) // 2 + 1):
            for i in range(len(step_names) - pattern_len * 2 + 1):
                pattern = step_names[i:i+pattern_len]
                next_seq = step_names[i+pattern_len:i+pattern_len*2]
                
                if pattern == next_seq:
                    return True
        
        return False
    
    def _detect_contradictions(self, state: ReasoningState) -> bool:
        """
        Detect contradictory statements in reasoning trace.
        
        Args:
            state: Current reasoning state
            
        Returns:
            True if contradictions detected, False otherwise
        """
        # Simple implementation: check if confidence oscillates dramatically
        if len(state.steps) < 3:
            return False
        
        # Extract confidence values
        confidences = []
        for step in state.steps:
            if "output" in step and "confidence" in step["output"]:
                confidences.append(step["output"]["confidence"])
        
        # Check for oscillations
        if len(confidences) >= 3:
            for i in range(len(confidences) - 2):
                # Check for significant up-down or down-up pattern
                if (confidences[i] - confidences[i+1] > 0.3 and confidences[i+1] - confidences[i+2] < -0.3) or \
                   (confidences[i] - confidences[i+1] < -0.3 and confidences[i+1] - confidences[i+2] > 0.3):
                    return True
        
        return False

    # Reflection utilities
    def fork_reflection(self, base_state: Dict[str, Any], depth: int = 2) -> List[Dict[str, Any]]:
        """
        Create multiple reflection paths from a base state.
        
        Args:
            base_state: Base state to fork from
            depth: Depth of reflection
            
        Returns:
            List of reflection results
        """
        reflection_paths = []
        
        # Create reflective dimensions
        dimensions = [
            "consistency",  # Consistency with philosophy
            "evidence",     # Evidence evaluation
            "alternatives", # Alternative hypotheses
            "biases",       # Cognitive biases
            "gaps",         # Knowledge gaps
        ]
        
        # Generate reflection for each dimension
        for dimension in dimensions:
            reflection = self._reflect_on_dimension(base_state, dimension, depth)
            reflection_paths.append({
                "dimension": dimension,
                "reflection": reflection,
                "confidence": reflection.get("confidence", 0.5),
            })
        
        # Sort by confidence (highest first)
        reflection_paths.sort(key=lambda x: x["confidence"], reverse=True)
        
        return reflection_paths
    
    def _reflect_on_dimension(self, state: Dict[str, Any], dimension: str, depth: int) -> Dict[str, Any]:
        """
        Reflect on a specific dimension of reasoning.
        
        Args:
            state: Current state
            dimension: Dimension to reflect on
            depth: Depth of reflection
            
        Returns:
            Reflection results
        """
        # Craft dimension-specific prompt
        if dimension == "consistency":
            prompt_focus = "the consistency between my decisions and my core philosophy"
        elif dimension == "evidence":
            prompt_focus = "the quality and completeness of evidence I'm considering"
        elif dimension == "alternatives":
            prompt_focus = "alternative hypotheses or viewpoints I might be overlooking"
        elif dimension == "biases":
            prompt_focus = "cognitive biases that might be affecting my judgment"
        elif dimension == "gaps":
            prompt_focus = "knowledge gaps that could be affecting my analysis"
        else:
            prompt_focus = f"the dimension of {dimension} in my reasoning"
        
        # Create reflection prompt
        prompt = f"""
        You are the {self.agent_name} agent with the following philosophy:
        "{self.agent_philosophy}"
        
        Perform a depth {depth} reflection focusing specifically on {prompt_focus}.
        
        Current state:
        {state}
        
        Respond with a JSON object containing:
        - dimension: "{dimension}"
        - assessment: Your assessment of this dimension
        - issues_identified: List of specific issues identified
        - recommendations: Recommendations to address these issues
        - confidence: Your confidence in this assessment (0-1)
        """
        
        try:
            # Generate reflection using LLM
            response = self.model_router.generate(prompt)
            
            # Parse JSON response (with fallback)
            try:
                import json
                reflection = json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing
                reflection = {
                    "dimension": dimension,
                    "assessment": "Unable to parse full reflection",
                    "issues_identified": [],
                    "recommendations": [],
                    "confidence": 0.3,
                }
            
            return reflection
            
        except Exception as e:
            logging.warning(f"Error in dimension reflection: {e}")
            return {
                "dimension": dimension,
                "assessment": "Reflection failed due to error",
                "issues_identified": ["reflection_failure"],
                "recommendations": ["error_handling"],
                "confidence": 0.2,
                "error": str(e),
            }
