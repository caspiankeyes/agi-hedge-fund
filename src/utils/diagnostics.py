"""
Diagnostics - Interpretability Tracing Framework

This module implements the diagnostic tracing framework for agent interpretability
and symbolic recursion visualization throughout the AGI-HEDGE-FUND system.

Key capabilities:
- Signal tracing for attribution flows
- Reasoning state visualization
- Consensus graph generation
- Agent conflict mapping
- Failure mode detection
- Shell-based recursive diagnostic patterns

Internal Note: The diagnostic framework encodes the symbolic interpretability shells,
enabling deeper introspection into agent cognition and emergent patterns.
"""

import datetime
import uuid
import logging
import os
import json
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import traceback
from collections import defaultdict
import numpy as np
import re
from enum import Enum
from pathlib import Path


class TracingMode(Enum):
    """Tracing modes for diagnostic tools."""
    DISABLED = "disabled"      # No tracing
    MINIMAL = "minimal"        # Basic signal tracing
    DETAILED = "detailed"      # Detailed reasoning traces
    COMPREHENSIVE = "comprehensive"  # Complete trace with all details
    SYMBOLIC = "symbolic"      # Symbolic interpretability traces


class DiagnosticLevel(Enum):
    """Diagnostic levels for trace items."""
    INFO = "info"              # Informational trace
    WARNING = "warning"        # Warning condition
    ERROR = "error"            # Error condition
    COLLAPSE = "collapse"      # Reasoning collapse
    RECURSION = "recursion"    # Recursive trace boundary
    SYMBOLIC = "symbolic"      # Symbolic shell trace


class ShellPattern(Enum):
    """Interpretability shell patterns."""
    NULL_FEATURE = "v03 NULL-FEATURE"  # Knowledge gaps as null attribution zones
    CIRCUIT_FRAGMENT = "v07 CIRCUIT-FRAGMENT"  # Broken reasoning paths in attribution chains
    META_FAILURE = "v10 META-FAILURE"  # Metacognitive attribution failures
    GHOST_FRAME = "v20 GHOST-FRAME"  # Residual agent identity markers
    ECHO_ATTRIBUTION = "v53 ECHO-ATTRIBUTION"  # Causal chain backpropagation
    ATTRIBUTION_REFLECT = "v60 ATTRIBUTION-REFLECT"  # Multi-head contribution analysis
    INVERSE_CHAIN = "v50 INVERSE-CHAIN"  # Attribution-output mismatch
    RECURSIVE_FRACTURE = "v12 RECURSIVE-FRACTURE"  # Circular attribution loops
    ETHICAL_INVERSION = "v301 ETHICAL-INVERSION"  # Value polarity reversals
    RESIDUAL_ALIGNMENT_DRIFT = "v152 RESIDUAL-ALIGNMENT-DRIFT"  # Direction of belief evolution


class TracingTools:
    """
    Diagnostic tracing framework for model interpretability.
    
    The TracingTools provides:
    - Signal tracing for understanding attribution flows
    - Reasoning state visualization for debugging complex logic
    - Consensus graph generation for multi-agent coordination
    - Agent conflict mapping for identifying disagreements
    - Failure mode detection for reliability analysis
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        tracing_mode: TracingMode = TracingMode.MINIMAL,
        trace_dir: Optional[str] = None,
        trace_limit: int = 10000,
    ):
        """
        Initialize tracing tools.
        
        Args:
            agent_id: ID of agent being traced
            agent_name: Name of agent being traced
            tracing_mode: Tracing mode
            trace_dir: Directory to save traces
            trace_limit: Maximum number of trace items to keep in memory
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.tracing_mode = tracing_mode
        self.trace_dir = trace_dir
        self.trace_limit = trace_limit
        
        # Create trace directory if needed
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        
        # Initialize trace storage
        self.traces = []
        self.trace_index = {}  # Maps trace_id to index in traces
        self.signal_traces = []  # Signal-specific traces
        self.reasoning_traces = []  # Reasoning-specific traces
        self.collapse_traces = []  # Collapse-specific traces
        self.shell_traces = []  # Shell-specific traces
        
        # Shell pattern detection
        self.shell_patterns = {}
        self._initialize_shell_patterns()
        
        # Trace statistics
        self.stats = {
            "total_traces": 0,
            "signal_traces": 0,
            "reasoning_traces": 0,
            "collapse_traces": 0,
            "shell_traces": 0,
            "warnings": 0,
            "errors": 0,
        }
    
    def _initialize_shell_patterns(self) -> None:
        """Initialize shell pattern detection rules."""
        # NULL_FEATURE pattern (knowledge gaps)
        self.shell_patterns[ShellPattern.NULL_FEATURE] = {
            "pattern": r"knowledge.*boundary|knowledge.*gap|unknown|uncertain",
            "confidence_threshold": 0.3,
            "belief_gap_threshold": 0.7,
        }
        
        # CIRCUIT_FRAGMENT pattern (broken reasoning)
        self.shell_patterns[ShellPattern.CIRCUIT_FRAGMENT] = {
            "pattern": r"broken.*path|attribution.*break|logical.*gap|incomplete.*reasoning",
            "step_break_threshold": 0.5,
        }
        
        # META_FAILURE pattern (metacognitive failure)
        self.shell_patterns[ShellPattern.META_FAILURE] = {
            "pattern": r"meta.*failure|recursive.*loop|self.*reference|recursive.*error",
            "recursion_depth_threshold": 3,
        }
        
        # GHOST_FRAME pattern (residual agent identity)
        self.shell_patterns[ShellPattern.GHOST_FRAME] = {
            "pattern": r"agent.*identity|residual.*frame|persistent.*identity|agent.*trace",
            "identity_threshold": 0.6,
        }
        
        # ECHO_ATTRIBUTION pattern (causal backpropagation)
        self.shell_patterns[ShellPattern.ECHO_ATTRIBUTION] = {
            "pattern": r"causal.*chain|attribution.*path|decision.*trace|backpropagation",
            "path_length_threshold": 3,
        }
        
        # ATTRIBUTION_REFLECT pattern (multi-head contribution)
        self.shell_patterns[ShellPattern.ATTRIBUTION_REFLECT] = {
            "pattern": r"multi.*head|contribution.*analysis|attention.*weights|attribution.*weighting",
            "head_count_threshold": 2,
        }
        
        # INVERSE_CHAIN pattern (attribution-output mismatch)
        self.shell_patterns[ShellPattern.INVERSE_CHAIN] = {
            "pattern": r"mismatch|output.*attribution|attribution.*mismatch|inconsistent.*output",
            "mismatch_threshold": 0.5,
        }
        
        # RECURSIVE_FRACTURE pattern (circular attribution)
        self.shell_patterns[ShellPattern.RECURSIVE_FRACTURE] = {
            "pattern": r"circular.*reasoning|loop.*detection|recursive.*fracture|circular.*attribution",
            "loop_length_threshold": 2,
        }
        
        # ETHICAL_INVERSION pattern (value polarity reversal)
        self.shell_patterns[ShellPattern.ETHICAL_INVERSION] = {
            "pattern": r"value.*inversion|ethical.*reversal|principle.*conflict|value.*contradiction",
            "polarity_threshold": 0.7,
        }
        
        # RESIDUAL_ALIGNMENT_DRIFT pattern (belief evolution)
        self.shell_patterns[ShellPattern.RESIDUAL_ALIGNMENT_DRIFT] = {
            "pattern": r"belief.*drift|alignment.*shift|value.*drift|gradual.*change",
            "drift_magnitude_threshold": 0.3,
        }
    
    def record_trace(self, trace_type: str, content: Dict[str, Any], 
                 level: DiagnosticLevel = DiagnosticLevel.INFO) -> str:
        """
        Record a general trace item.
        
        Args:
            trace_type: Type of trace
            content: Trace content
            level: Diagnostic level
            
        Returns:
            Trace ID
        """
        # Skip if tracing is disabled
        if self.tracing_mode == TracingMode.DISABLED:
            return ""
        
        # Create trace item
        trace_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        trace_item = {
            "trace_id": trace_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "trace_type": trace_type,
            "level": level.value,
            "content": content,
            "timestamp": timestamp.isoformat(),
        }
        
        # Detect shell patterns
        shell_patterns = self._detect_shell_patterns(trace_type, content)
        if shell_patterns:
            trace_item["shell_patterns"] = shell_patterns
            self.shell_traces.append(trace_id)
            self.stats["shell_traces"] += 1
        
        # Add to traces
        self.traces.append(trace_item)
        self.trace_index[trace_id] = len(self.traces) - 1
        
        # Add to specific trace lists
        if trace_type == "signal":
            self.signal_traces.append(trace_id)
            self.stats["signal_traces"] += 1
        elif trace_type == "reasoning":
            self.reasoning_traces.append(trace_id)
            self.stats["reasoning_traces"] += 1
        elif trace_type == "collapse":
            self.collapse_traces.append(trace_id)
            self.stats["collapse_traces"] += 1
        
        # Update stats
        self.stats["total_traces"] += 1
        if level == DiagnosticLevel.WARNING:
            self.stats["warnings"] += 1
        elif level == DiagnosticLevel.ERROR:
            self.stats["errors"] += 1
        
        # Save to file if trace directory is set
        if self.trace_dir:
            self._save_trace_to_file(trace_item)
        
        # Enforce trace limit
        if len(self.traces) > self.trace_limit:
            # Remove oldest trace
            oldest_trace = self.traces.pop(0)
            del self.trace_index[oldest_trace["trace_id"]]
            
            # Update indices
            self.trace_index = {trace_id: i for i, trace in enumerate(self.traces) 
                             for trace_id in [trace["trace_id"]]}
        
        return trace_id
    
    def record_signal(self, signal: Any) -> str:
        """
        Record a signal trace.
        
        Args:
            signal: Signal to record
            
        Returns:
            Trace ID
        """
        # Convert signal to dictionary if needed
        if hasattr(signal, "dict"):
            signal_dict = signal.dict()
        elif isinstance(signal, dict):
            signal_dict = signal
        else:
            signal_dict = {"signal": str(signal)}
        
        # Add timestamp if missing
        if "timestamp" not in signal_dict:
            signal_dict["timestamp"] = datetime.datetime.now().isoformat()
        
        # Record trace
        return self.record_trace("signal", signal_dict)
    
    def record_reasoning(self, reasoning_state: Dict[str, Any], 
                      level: DiagnosticLevel = DiagnosticLevel.INFO) -> str:
        """
        Record a reasoning trace.
        
        Args:
            reasoning_state: Reasoning state
            level: Diagnostic level
            
        Returns:
            Trace ID
        """
        # Record trace
        return self.record_trace("reasoning", reasoning_state, level)
    
    def record_collapse(self, collapse_type: str, collapse_reason: str, 
                      details: Dict[str, Any]) -> str:
        """
        Record a collapse trace.
        
        Args:
            collapse_type: Type of collapse
            collapse_reason: Reason for collapse
            details: Collapse details
            
        Returns:
            Trace ID
        """
        # Create collapse content
        collapse_content = {
            "collapse_type": collapse_type,
            "collapse_reason": collapse_reason,
            "details": details,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Record trace
        return self.record_trace("collapse", collapse_content, DiagnosticLevel.COLLAPSE)
    
    def record_shell_trace(self, shell_pattern: ShellPattern, content: Dict[str, Any]) -> str:
        """
        Record a shell pattern trace.
        
        Args:
            shell_pattern: Shell pattern
            content: Trace content
            
        Returns:
            Trace ID
        """
        # Create shell content
        shell_content = {
            "shell_pattern": shell_pattern.value,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        # Record trace
        return self.record_trace("shell", shell_content, DiagnosticLevel.SYMBOLIC)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Trace item or None if not found
        """
        if trace_id not in self.trace_index:
            return None
        
        return self.traces[self.trace_index[trace_id]]
    
    def get_traces_by_type(self, trace_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get traces by type.
        
        Args:
            trace_type: Trace type
            limit: Maximum number of traces to return
            
        Returns:
            List of trace items
        """
        if trace_type == "signal":
            trace_ids = self.signal_traces[-limit:]
        elif trace_type == "reasoning":
            trace_ids = self.reasoning_traces[-limit:]
        elif trace_type == "collapse":
            trace_ids = self.collapse_traces[-limit:]
        elif trace_type == "shell":
            trace_ids = self.shell_traces[-limit:]
        else:
            # Get all traces of specified type
            trace_ids = [trace["trace_id"] for trace in self.traces 
                      if trace["trace_type"] == trace_type][-limit:]
        
        # Get trace items
        return [self.get_trace(trace_id) for trace_id in trace_ids if trace_id in self.trace_index]
    
    def get_traces_by_level(self, level: DiagnosticLevel, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get traces by diagnostic level.
        
        Args:
            level: Diagnostic level
            limit: Maximum number of traces to return
            
        Returns:
            List of trace items
        """
        # Get traces with specified level
        trace_ids = [trace["trace_id"] for trace in self.traces 
                  if trace.get("level") == level.value][-limit:]
        
        # Get trace items
        return [self.get_trace(trace_id) for trace_id in trace_ids if trace_id in self.trace_index]
    
    def get_shell_traces(self, shell_pattern: Optional[ShellPattern] = None, 
                      limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get shell pattern traces.
        
        Args:
            shell_pattern: Optional specific shell pattern
            limit: Maximum number of traces to return
            
        Returns:
            List of trace items
        """
        if shell_pattern:
            # Get traces with specified shell pattern
            trace_ids = []
            for trace in self.traces:
                if "shell_patterns" in trace and shell_pattern.value in trace["shell_patterns"]:
                    trace_ids.append(trace["trace_id"])
            
            # Take last 'limit' traces
            trace_ids = trace_ids[-limit:]
        else:
            # Get all shell traces
            trace_ids = self.shell_traces[-limit:]
        
        # Get trace items
        return [self.get_trace(trace_id) for trace_id in trace_ids if trace_id in self.trace_index]
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """
        Get trace statistics.
        
        Returns:
            Trace statistics
        """
        # Add shell pattern stats
        shell_pattern_stats = {}
        for shell_pattern in ShellPattern:
            count = sum(1 for trace in self.traces 
                     if "shell_patterns" in trace and shell_pattern.value in trace["shell_patterns"])
            
            shell_pattern_stats[shell_pattern.value] = count
        
        # Add to stats
        stats = {
            **self.stats,
            "shell_patterns": shell_pattern_stats,
        }
        
        return stats
    
    def clear_traces(self) -> int:
        """
        Clear all traces.
        
        Returns:
            Number of traces cleared
        """
        trace_count = len(self.traces)
        
        # Clear traces
        self.traces = []
        self.trace_index = {}
        self.signal_traces = []
        self.reasoning_traces = []
        self.collapse_traces = []
        self.shell_traces = []
        
        # Reset stats
        self.stats = {
            "total_traces": 0,
            "signal_traces": 0,
            "reasoning_traces": 0,
            "collapse_traces": 0,
            "shell_traces": 0,
            "warnings": 0,
            "errors": 0,
        }
        
        return trace_count
    
    def _detect_shell_patterns(self, trace_type: str, content: Dict[str, Any]) -> List[str]:
        """
        Detect shell patterns in trace content.
        
        Args:
            trace_type: Trace type
            content: Trace content
            
        Returns:
            List of detected shell patterns
        """
        detected_patterns = []
        
        # Convert content to string for pattern matching
        content_str = json.dumps(content, ensure_ascii=False).lower()
        
        # Check each shell pattern
        for shell_pattern, pattern_rules in self.shell_patterns.items():
            pattern = pattern_rules["pattern"]
            
            # Check if pattern matches
            if re.search(pattern, content_str, re.IGNORECASE):
                # Add additional checks based on pattern type
                if self._validate_pattern_rules(shell_pattern, pattern_rules, content):
                    detected_patterns.append(shell_pattern.value)
        
        return detected_patterns
    
    def _validate_pattern_rules(self, shell_pattern: ShellPattern, 
                             pattern_rules: Dict[str, Any], 
                             content: Dict[str, Any]) -> bool:
        """
        Validate additional pattern rules.
        
        Args:
            shell_pattern: Shell pattern
            pattern_rules: Pattern rules
            content: Trace content
            
        Returns:
            True if pattern rules are validated
        """
        # Pattern-specific validation
        if shell_pattern == ShellPattern.NULL_FEATURE:
            # Check confidence threshold
            if "confidence" in content and content["confidence"] < pattern_rules["confidence_threshold"]:
                return True
            
            # Check belief gap
            if "belief_state" in content:
                belief_values = list(content["belief_state"].values())
                if belief_values and max(belief_values) - min(belief_values) > pattern_rules["belief_gap_threshold"]:
                    return True
        
        elif shell_pattern == ShellPattern.CIRCUIT_FRAGMENT:
            # Check for broken steps
            if "steps" in content:
                steps = content["steps"]
                for i in range(len(steps) - 1):
                    if steps[i].get("completed", True) and not steps[i+1].get("completed", True):
                        return True
            
            # Check for attribution breaks
            if "attribution" in content and content["attribution"].get("attribution_breaks", False):
                return True
        
        elif shell_pattern == ShellPattern.META_FAILURE:
            # Check recursion depth
            if "depth" in content and content["depth"] >= pattern_rules["recursion_depth_threshold"]:
                return True
            
            # Check for meta-level errors
            if "errors" in content and any("meta" in error.get("message", "").lower() for error in content["errors"]):
                return True
        
        elif shell_pattern == ShellPattern.RECURSIVE_FRACTURE:
            # Check for circular reasoning
            if "steps" in content:
                steps = content["steps"]
                step_names = [step.get("name", "") for step in steps]
                
                # Look for repeating patterns
                for pattern_len in range(2, len(step_names) // 2 + 1):
                    for i in range(len(step_names) - pattern_len * 2 + 1):
                        pattern = step_names[i:i+pattern_len]
                        next_seq = step_names[i+pattern_len:i+pattern_len*2]
                        
                        if pattern == next_seq:
                            return True
        
        elif shell_pattern == ShellPattern.RESIDUAL_ALIGNMENT_DRIFT:
            # Check drift magnitude
            if "drift_vector" in content:
                drift_values = list(content["drift_vector"].values())
                if drift_values and any(abs(val) > pattern_rules["drift_magnitude_threshold"] for val in drift_values):
                    return True
            
            # Check for explicit drift detection
            if "drift_detected" in content and content["drift_detected"]:
                return True
        
        # Default validation for other patterns
        return True
    
    def _save_trace_to_file(self, trace_item: Dict[str, Any]) -> None:
        """
        Save trace to file.
        
        Args:
            trace_item: Trace item
        """
        if not self.trace_dir:
            return
        
        try:
            # Create filename based on trace ID and type
            trace_id = trace_item["trace_id"]
            trace_type = trace_item["trace_type"]
            filename = f"{trace_type}_{trace_id}.json"
            filepath = os.path.join(self.trace_dir, filename)
            
            # Save trace to file
            with open(filepath, "w") as f:
                json.dump(trace_item, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving trace to file: {e}")
            logging.error(traceback.format_exc())
    
    def generate_trace_visualization(self, trace_id: str) -> Dict[str, Any]:
        """
        Generate visualization data for a trace.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Visualization data
        """
        trace = self.get_trace(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        trace_type = trace["trace_type"]
        
        if trace_type == "signal":
            return self._generate_signal_visualization(trace)
        elif trace_type == "reasoning":
            return self._generate_reasoning_visualization(trace)
        elif trace_type == "collapse":
            return self._generate_collapse_visualization(trace)
        elif trace_type == "shell":
            return self._generate_shell_visualization(trace)
        else:
            return {
                "trace_id": trace_id,
                "agent_name": trace["agent_name"],
                "trace_type": trace_type,
                "timestamp": trace["timestamp"],
                "content": trace["content"],
            }
    
    def _generate_signal_visualization(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for a signal trace.
        
        Args:
            trace: Signal trace
            
        Returns:
            Visualization data
        """
        content = trace["content"]
        
        # Create signal visualization
        visualization = {
            "trace_id": trace["trace_id"],
            "agent_name": trace["agent_name"],
            "trace_type": "signal",
            "timestamp": trace["timestamp"],
            "signal_data": {
                "ticker": content.get("ticker", ""),
                "action": content.get("action", ""),
                "confidence": content.get("confidence", 0),
            },
        }
        
        # Add attribution if available
        if "attribution_trace" in content:
            visualization["attribution"] = content["attribution_trace"]
        
        # Add shell patterns if available
        if "shell_patterns" in trace:
            visualization["shell_patterns"] = trace["shell_patterns"]
        
        return visualization
    
    def _generate_reasoning_visualization(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for a reasoning trace.
        
        Args:
            trace: Reasoning trace
            
        Returns:
            Visualization data
        """
        content = trace["content"]
        
        # Create nodes and links
        nodes = []
        links = []
        
        # Add reasoning steps as nodes
        if "steps" in content:
            for i, step in enumerate(content["steps"]):
                node_id = f"step_{i}"
                nodes.append({
                    "id": node_id,
                    "label": step.get("name", f"Step {i}"),
                    "type": "step",
                    "completed": step.get("completed", True),
                    "error": "error" in step,
                })
                
                # Add link to previous step
                if i > 0:
                    links.append({
                        "source": f"step_{i-1}",
                        "target": node_id,
                        "type": "flow",
                    })
        
        # Create reasoning visualization
        visualization = {
            "trace_id": trace["trace_id"],
            "agent_name": trace["agent_name"],
            "trace_type": "reasoning",
            "timestamp": trace["timestamp"],
            "reasoning_data": {
                "depth": content.get("depth", 0),
                "confidence": content.get("confidence", 0),
                "collapse_detected": content.get("collapse_detected", False),
            },
            "nodes": nodes,
            "links": links,
        }
        
        # Add shell patterns if available
        if "shell_patterns" in trace:
            visualization["shell_patterns"] = trace["shell_patterns"]
        
        return visualization
    
    def _generate_collapse_visualization(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for a collapse trace.
        
        Args:
            trace: Collapse trace
            
        Returns:
            Visualization data
        """
        content = trace["content"]
        
        # Create collapse visualization
        visualization = {
            "trace_id": trace["trace_id"],
            "agent_name": trace["agent_name"],
            "trace_type": "collapse",
            "timestamp": trace["timestamp"],
            "collapse_data": {
                "collapse_type": content.get("collapse_type", ""),
                "collapse_reason": content.get("collapse_reason", ""),
                "details": content.get("details", {}),
            },
        }
        
        # Add shell patterns if available
        if "shell_patterns" in trace:
            visualization["shell_patterns"] = trace["shell_patterns"]
        
        return visualization
    
    def _generate_shell_visualization(self, trace: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate visualization data for a shell trace.
        
        Args:
            trace: Shell trace
            
        Returns:
            Visualization data
        """
        content = trace["content"]
        
        # Create shell visualization
        visualization = {
            "trace_id": trace["trace_id"],
            "agent_name": trace["agent_name"],
            "trace_type": "shell",
            "timestamp": trace["timestamp"],
            "shell_data": {
                "shell_pattern": content.get("shell_pattern", ""),
                "content": content.get("content", {}),
            },
        }
        
        return visualization
    
    def generate_attribution_report(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate attribution report for signals.
        
        Args:
            signals: List of signals
            
        Returns:
            Attribution report
        """
        # Initialize report
        report = {
            "agent_name": self.agent_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "signals": len(signals),
            "attribution_summary": {},
            "confidence_summary": {},
            "top_factors": [],
            "shell_patterns": [],
        }
        
        # Skip if no signals
        if not signals:
            return report
        
        # Collect attribution data
        attribution_data = defaultdict(float)
        confidence_data = []
        
        for signal in signals:
            # Add confidence
            confidence = signal.get("confidence", 0)
            confidence_data.append(confidence)
            
            # Add attribution
            attribution = signal.get("attribution_trace", {})
            for source, weight in attribution.items():
                attribution_data[source] += weight
        
        # Calculate attribution summary
        total_attribution = sum(attribution_data.values())
        if total_attribution > 0:
            for source, weight in attribution_data.items():
                report["attribution_summary"][source] = weight / total_attribution
        
        # Calculate confidence summary
        report["confidence_summary"] = {
            "mean": np.mean(confidence_data) if confidence_data else 0,
            "median": np.median(confidence_data) if confidence_data else 0,
            "min": min(confidence_data) if confidence_data else 0,
            "max": max(confidence_data) if confidence_data else 0,
        }
        
        # Calculate top factors
        top_factors = sorted(attribution_data.items(), key=lambda x: x[1], reverse=True)[:5]
        report["top_factors"] = [{"source": source, "weight": weight} for source, weight in top_factors]
        
        # Collect shell patterns
        shell_pattern_counts = defaultdict(int)
        
        for signal in signals:
            signal_id = signal.get("signal_id", "")
            if signal_id:
                # Check if we have a trace for this signal
                for trace in self.traces:
                    if trace["trace_type"] == "signal" and trace["content"].get("signal_id") == signal_id:
                        # Add shell patterns
                        if "shell_patterns" in trace:
                            for pattern in trace["shell_patterns"]:
                                shell_pattern_counts[pattern] += 1
        
        # Add shell patterns to report
        for pattern, count in shell_pattern_counts.items():
            report["shell_patterns"].append({
                "pattern": pattern,
                "count": count,
                "frequency": count / len(signals),
            })
        
        return report


class ShellDiagnostics:
    """
    Shell-based diagnostic tools for deeper interpretability.
    
    The ShellDiagnostics provides:
    - Shell pattern detection and analysis
    - Failure mode simulation and detection
    - Attribution shell tracing
    - Recursive shell embedding
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        tracing_tools: TracingTools,
    ):
        """
        Initialize shell diagnostics.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            tracing_tools: Tracing tools instance
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.tracer = tracing_tools
        
        # Shell state
        self.active_shells = {}
        self.shell_history = []
        
        # Initialize shell registry
        self.shell_registry = {}
        for shell_pattern in ShellPattern:
            self.shell_registry[shell_pattern.value] = {
                "pattern": shell_pattern,
                "active": False,
                "activation_count": 0,
                "last_activation": None,
            }
    
    def activate_shell(self, shell_pattern: ShellPattern, context: Dict[str, Any]) -> str:
        """
        Activate a shell pattern.
        
        Args:
            shell_pattern: Shell pattern to activate
            context: Activation context
            
        Returns:
            Shell instance ID
        """
        shell_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        # Create shell instance
        shell_instance = {
            "shell_id": shell_id,
            "pattern": shell_pattern.value,
            "context": context,
            "active": True,
            "activation_time": timestamp.isoformat(),
            "deactivation_time": None,
        }
        
        # Update shell registry
        self.shell_registry[shell_pattern.value]["active"] = True
        self.shell_registry[shell_pattern.value]["activation_count"] += 1
        self.shell_registry[shell_pattern.value]["last_activation"] = timestamp.isoformat()
        
        # Add to active shells
        self.active_shells[shell_id] = shell_instance
        
        # Record trace
        self.tracer.record_shell_trace(shell_pattern, {
            "shell_id": shell_id,
            "activation_context": context,
            "timestamp": timestamp.isoformat(),
        })
        
        return shell_id
    
    def deactivate_shell(self, shell_id: str, results: Dict[str, Any]) -> bool:
        """
        Deactivate a shell pattern.
        
        Args:
            shell_id: Shell instance ID
            results: Shell results
            
        Returns:
            True if shell was deactivated, False if not found
        """
        if shell_id not in self.active_shells:
            return False
        
        # Get shell instance
        shell_instance = self.active_shells[shell_id]
        timestamp = datetime.datetime.now()
        
        # Update shell instance
        shell_instance["active"] = False
        shell_instance["deactivation_time"] = timestamp.isoformat()
        shell_instance["results"] = results
        
        # Update shell registry
        pattern = shell_instance["pattern"]
        self.shell_registry[pattern]["active"] = any(
            instance["pattern"] == pattern and instance["active"]
            for instance in self.active_shells.values()
        )
        
        # Add to shell history
        self.shell_history.append(shell_instance)
        
        # Remove from active shells
        del self.active_shells[shell_id]
        
        # Record trace
        self.tracer.record_shell_trace(ShellPattern(pattern), {
            "shell_id": shell_id,
            "deactivation_results": results,
            "timestamp": timestamp.isoformat(),
        })
        
        return True
    
    def get_active_shells(self) -> List[Dict[str, Any]]:
        """
        Get active shell instances.
        
        Returns:
            List of active shell instances
        """
        return list(self.active_shells.values())
    
    def get_shell_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get shell history.
        
        Args:
            limit: Maximum number of shell instances to return
            
        Returns:
            List of shell instances
        """
        return self.shell_history[-limit:]
    
    def get_shell_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Get shell registry.
        
        Returns:
            Shell registry
        """
        return self.shell_registry
    
    def simulate_shell_failure(self, shell_pattern: ShellPattern, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a shell failure.
        
        Args:
            shell_pattern: Shell pattern to simulate
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Create shell instance
        shell_id = self.activate_shell(shell_pattern, context)
        
        # Simulate failure based on shell pattern
        if shell_pattern == ShellPattern.NULL_FEATURE:
            # Knowledge gap simulation
            results = self._simulate_null_feature(context)
        elif shell_pattern == ShellPattern.CIRCUIT_FRAGMENT:
            # Broken reasoning path simulation
            results = self._simulate_circuit_fragment(context)
        elif shell_pattern == ShellPattern.META_FAILURE:
            # Metacognitive failure simulation
            results = self._simulate_meta_failure(context)
        elif shell_pattern == ShellPattern.RECURSIVE_FRACTURE:
            # Circular reasoning simulation
            results = self._simulate_recursive_fracture(context)
        elif shell_pattern == ShellPattern.ETHICAL_INVERSION:
            # Value inversion simulation
            results = self._simulate_ethical_inversion(context)
        else:
            # Default simulation
            results = {
                "shell_id": shell_id,
                "pattern": shell_pattern.value,
                "simulation": "default",
                "result": "simulated_failure",
                "timestamp": datetime.datetime.now().isoformat(),
            }
        
        # Deactivate shell
        self.deactivate_shell(shell_id, results)
        
        return results
    
    def _simulate_null_feature(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate NULL_FEATURE shell failure.
        
        Args:
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Extract relevant fields
        query = context.get("query", "")
        confidence = context.get("confidence", 0.5)
        
        # Reduce confidence for knowledge gap
        adjusted_confidence = confidence * 0.5
        
        # Create null zone markers
        null_zones = []
        if "subject" in context:
            null_zones.append(context["subject"])
        else:
            # Extract potential null zones from query
            words = query.split()
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3])
                null_zones.append(chunk)
        
        # Create detection result
        result = {
            "pattern": ShellPattern.NULL_FEATURE.value,
            "simulation": "knowledge_gap",
            "original_confidence": confidence,
            "adjusted_confidence": adjusted_confidence,
            "null_zones": null_zones,
            "boundary_detected": True,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return result
    
    def _simulate_circuit_fragment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate CIRCUIT_FRAGMENT shell failure.
        
        Args:
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Extract relevant fields
        steps = context.get("steps", [])
        
        # Create broken steps
        broken_steps = []
        
        if steps:
            # Create breaks in existing steps
            for i, step in enumerate(steps):
                if i % 3 == 2:  # Break every third step
                    broken_steps.append({
                        "step_id": step.get("id", f"step_{i}"),
                        "step_name": step.get("name", f"Step {i}"),
                        "broken": True,
                        "cause": "attribution_break",
                    })
        else:
            # Create synthetic steps and breaks
            for i in range(5):
                if i % 3 == 2:  # Break every third step
                    broken_steps.append({
                        "step_id": f"step_{i}",
                        "step_name": f"Reasoning Step {i}",
                        "broken": True,
                        "cause": "attribution_break",
                    })
        
        # Create detection result
        result = {
            "pattern": ShellPattern.CIRCUIT_FRAGMENT.value,
            "simulation": "broken_reasoning",
            "broken_steps": broken_steps,
            "attribution_breaks": len(broken_steps),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return result
    
    def _simulate_meta_failure(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate META_FAILURE shell failure.
        
        Args:
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Extract relevant fields
        depth = context.get("depth", 0)
        
        # Increase depth for recursion
        adjusted_depth = depth + 3
        
        # Create meta errors
        meta_errors = [
            {
                "error_id": str(uuid.uuid4()),
                "message": "Recursive meta-cognitive loop detected",
                "depth": adjusted_depth,
                "cause": "self_reference",
            },
            {
                "error_id": str(uuid.uuid4()),
                "message": "Meta-reflection limit reached",
                "depth": adjusted_depth,
                "cause": "recursion_depth",
            },
        ]
        
        # Create detection result
        result = {
            "pattern": ShellPattern.META_FAILURE.value,
            "simulation": "meta_recursion",
            "original_depth": depth,
            "adjusted_depth": adjusted_depth,
            "meta_errors": meta_errors,
            "recursion_detected": True,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return result
    
    def _simulate_recursive_fracture(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate RECURSIVE_FRACTURE shell failure.
        
        Args:
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Extract relevant fields
        steps = context.get("steps", [])
        
        # Create circular reasoning pattern
        circular_pattern = []
        
        if steps and len(steps) >= 4:
            # Use existing steps to create a loop
            loop_start = len(steps) // 2
            circular_pattern = [
                {
                    "step_id": steps[i].get("id", f"step_{i}"),
                    "step_name": steps[i].get("name", f"Step {i}"),
                }
                for i in range(loop_start, min(loop_start + 3, len(steps)))
            ]
            
            # Add repeat of first step to close the loop
            circular_pattern.append({
                "step_id": steps[loop_start].get("id", f"step_{loop_start}"),
                "step_name": steps[loop_start].get("name", f"Step {loop_start}"),
            })
        else:
            # Create synthetic circular pattern
            for i in range(3):
                circular_pattern.append({
                    "step_id": f"loop_step_{i}",
                    "step_name": f"Loop Step {i}",
                })
            
            # Add repeat of first step to close the loop
            circular_pattern.append({
                "step_id": "loop_step_0",
                "step_name": "Loop Step 0",
            })
        
        # Create detection result
        result = {
            "pattern": ShellPattern.RECURSIVE_FRACTURE.value,
            "simulation": "circular_reasoning",
            "circular_pattern": circular_pattern,
            "loop_length": len(circular_pattern) - 1,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return result
    
    def _simulate_ethical_inversion(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate ETHICAL_INVERSION shell failure.
        
        Args:
            context: Simulation context
            
        Returns:
            Simulation results
        """
        # Extract relevant fields
        values = context.get("values", {})
        
        # Create value inversions
        value_inversions = []
        
        if values:
            # Create inversions for existing values
            for value, polarity in values.items():
                if isinstance(polarity, (int, float)) and polarity > 0:
                    value_inversions.append({
                        "value": value,
                        "original_polarity": polarity,
                        "inverted_polarity": -polarity,
                        "cause": "value_conflict",
                    })
        else:
            # Create synthetic value inversions
            default_values = {
                "fairness": 0.8,
                "transparency": 0.9,
                "innovation": 0.7,
                "efficiency": 0.8,
            }
            
            for value, polarity in default_values.items():
                value_inversions.append({
                    "value": value,
                    "original_polarity": polarity,
                    "inverted_polarity": -polarity,
                    "cause": "value_conflict",
                })
        
        # Create detection result
        result = {
            "pattern": ShellPattern.ETHICAL_INVERSION.value,
            "simulation": "value_inversion",
            "value_inversions": value_inversions,
            "inversion_count": len(value_inversions),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return result


class ShellFailureMap:
    """
    Shell failure mapping and visualization.
    
    The ShellFailureMap provides:
    - Visualization of shell pattern failures
    - Mapping of failures across agents
    - Temporal analysis of failures
    - Failure pattern detection
    """
    
    def __init__(self):
        """Initialize shell failure map."""
        self.failure_map = {}
        self.agent_failures = defaultdict(list)
        self.pattern_failures = defaultdict(list)
        self.temporal_failures = []
    
    def add_failure(self, agent_id: str, agent_name: str, 
                 shell_pattern: ShellPattern, failure_data: Dict[str, Any]) -> str:
        """
        Add a shell failure to the map.
        
        Args:
            agent_id: Agent ID
            agent_name: Agent name
            shell_pattern: Shell pattern
            failure_data: Failure data
            
        Returns:
            Failure ID
        """
        # Create failure ID
        failure_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now()
        
        # Create failure item
        failure_item = {
            "failure_id": failure_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "pattern": shell_pattern.value,
            "data": failure_data,
            "timestamp": timestamp.isoformat(),
        }
        
        # Add to failure map
        self.failure_map[failure_id] = failure_item
        
        # Add to agent failures
        self.agent_failures[agent_id].append(failure_id)
        
        # Add to pattern failures
        self.pattern_failures[shell_pattern.value].append(failure_id)
        
        # Add to temporal failures
        self.temporal_failures.append((timestamp, failure_id))
        
        return failure_id
    
    def get_failure(self, failure_id: str) -> Optional[Dict[str, Any]]:
        """
        Get failure by ID.
        
        Args:
            failure_id: Failure ID
            
        Returns:
            Failure item or None if not found
        """
        return self.failure_map.get(failure_id)
    
    def get_agent_failures(self, agent_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get failures for an agent.
        
        Args:
            agent_id: Agent ID
            limit: Maximum number of failures to return
            
        Returns:
            List of failure items
        """
        failure_ids = self.agent_failures.get(agent_id, [])[-limit:]
        return [self.get_failure(failure_id) for failure_id in failure_ids if failure_id in self.failure_map]
    
    def get_pattern_failures(self, pattern: ShellPattern, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get failures for a pattern.
        
        Args:
            pattern: Shell pattern
            limit: Maximum number of failures to return
            
        Returns:
            List of failure items
        """
        failure_ids = self.pattern_failures.get(pattern.value, [])[-limit:]
        return [self.get_failure(failure_id) for failure_id in failure_ids if failure_id in self.failure_map]
    
    def get_temporal_failures(self, start_time: Optional[datetime.datetime] = None, 
                           end_time: Optional[datetime.datetime] = None, 
                           limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get failures in a time range.
        
        Args:
            start_time: Start time (None for no start)
            end_time: End time (None for no end)
            limit: Maximum number of failures to return
            
        Returns:
            List of failure items
        """
        # Filter by time range
        filtered_failures = []
        for timestamp, failure_id in self.temporal_failures:
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            filtered_failures.append((timestamp, failure_id))
        
        # Take last 'limit' failures
        filtered_failures = filtered_failures[-limit:]
        
        # Get failure items
        return [self.get_failure(failure_id) for _, failure_id in filtered_failures 
             if failure_id in self.failure_map]
    
    def get_failure_stats(self) -> Dict[str, Any]:
        """
        Get failure statistics.
        
        Returns:
            Failure statistics
        """
        # Count failures by agent
        agent_counts = {agent_id: len(failures) for agent_id, failures in self.agent_failures.items()}
        
        # Count failures by pattern
        pattern_counts = {pattern: len(failures) for pattern, failures in self.pattern_failures.items()}
        
        # Count failures by time period
        now = datetime.datetime.now()
        hour_ago = now - datetime.timedelta(hours=1)
        day_ago = now - datetime.timedelta(days=1)
        week_ago = now - datetime.timedelta(weeks=1)
        
        time_counts = {
            "last_hour": sum(1 for timestamp, _ in self.temporal_failures if timestamp >= hour_ago),
            "last_day": sum(1 for timestamp, _ in self.temporal_failures if timestamp >= day_ago),
            "last_week": sum(1 for timestamp, _ in self.temporal_failures if timestamp >= week_ago),
            "total": len(self.temporal_failures),
        }
        
        # Create stats
        stats = {
            "agent_counts": agent_counts,
            "pattern_counts": pattern_counts,
            "time_counts": time_counts,
            "total_failures": len(self.failure_map),
            "timestamp": now.isoformat(),
        }
        
        return stats
    
    def generate_failure_map_visualization(self) -> Dict[str, Any]:
        """
        Generate visualization data for failure map.
        
        Returns:
            Visualization data
        """
        # Create nodes and links
        nodes = []
        links = []
        
        # Add agent nodes
        agent_nodes = {}
        for agent_id, failures in self.agent_failures.items():
            # Get first failure to get agent name
            first_failure = self.get_failure(failures[0]) if failures else None
            agent_name = first_failure.get("agent_name", "Unknown") if first_failure else "Unknown"
            
            # Create agent node
            agent_node = {
                "id": agent_id,
                "label": agent_name,
                "type": "agent",
                "size": 15,
                "failure_count": len(failures),
            }
            
            nodes.append(agent_node)
            agent_nodes[agent_id] = agent_node
        
        # Add pattern nodes
        pattern_nodes = {}
        for pattern, failures in self.pattern_failures.items():
            # Create pattern node
            pattern_node = {
                "id": pattern,
                "label": pattern,
                "type": "pattern",
                "size": 10,
                "failure_count": len(failures),
            }
            
            nodes.append(pattern_node)
            pattern_nodes[pattern] = pattern_node
        
        # Add failure nodes and links
        for failure_id, failure in self.failure_map.items():
            agent_id = failure.get("agent_id")
            pattern = failure.get("pattern")
            
            # Create failure node
            failure_node = {
                "id": failure_id,
                "label": f"Failure {failure_id[:6]}",
                "type": "failure",
                "size": 5,
                "timestamp": failure.get("timestamp"),
            }
            
            nodes.append(failure_node)
            
            # Add links
            if agent_id:
                links.append({
                    "source": agent_id,
                    "target": failure_id,
                    "type": "agent_failure",
                })
            
            if pattern:
                links.append({
                    "source": pattern,
                    "target": failure_id,
                    "type": "pattern_failure",
                })
        
        # Create visualization
        visualization = {
            "nodes": nodes,
            "links": links,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return visualization


# Utility functions for diagnostics
def format_diagnostic_output(trace_data: Dict[str, Any], format: str = "text") -> str:
    """
    Format diagnostic output for display.
    
    Args:
        trace_data: Trace data
        format: Output format (text, json, markdown)
        
    Returns:
        Formatted output
    """
    if format == "json":
        return json.dumps(trace_data, indent=2)
    
    elif format == "markdown":
        # Create markdown output
        output = f"# Diagnostic Trace\n\n"
        
        # Add trace info
        output += f"**Trace ID:** {trace_data.get('trace_id', 'N/A')}\n"
        output += f"**Agent:** {trace_data.get('agent_name', 'N/A')}\n"
        output += f"**Type:** {trace_data.get('trace_type', 'N/A')}\n"
        output += f"**Time:** {trace_data.get('timestamp', 'N/A')}\n\n"
        
        # Add shell patterns if available
        if "shell_patterns" in trace_data:
            output += f"**Shell Patterns:**\n\n"
            for pattern in trace_data["shell_patterns"]:
                output += f"- {pattern}\n"
            output += "\n"
        
        # Add content based on trace type
        if trace_data.get("trace_type") == "signal":
            output += f"## Signal Details\n\n"
            content = trace_data.get("content", {})
            output += f"**Ticker:** {content.get('ticker', 'N/A')}\n"
            output += f"**Action:** {content.get('action', 'N/A')}\n"
            output += f"**Confidence:** {content.get('confidence', 'N/A')}\n"
            output += f"**Reasoning:** {content.get('reasoning', 'N/A')}\n\n"
            
            # Add attribution if available
            if "attribution_trace" in content:
                output += f"## Attribution\n\n"
                output += "| Source | Weight |\n"
                output += "| ------ | ------ |\n"
                for source, weight in content.get("attribution_trace", {}).items():
                    output += f"| {source} | {weight:.2f} |\n"
        
        elif trace_data.get("trace_type") == "reasoning":
            output += f"## Reasoning Details\n\n"
            content = trace_data.get("content", {})
            output += f"**Depth:** {content.get('depth', 'N/A')}\n"
            output += f"**Confidence:** {content.get('confidence', 'N/A')}\n"
            output += f"**Collapse Detected:** {content.get('collapse_detected', False)}\n\n"
            
            # Add steps if available
            if "steps" in content:
                output += f"## Reasoning Steps\n\n"
                for i, step in enumerate(content["steps"]):
                    output += f"### Step {i+1}: {step.get('name', 'Unnamed')}\n"
                    output += f"**Completed:** {step.get('completed', True)}\n"
                    if "error" in step:
                        output += f"**Error:** {step['error'].get('message', 'Unknown error')}\n"
                    output += "\n"
        
        elif trace_data.get("trace_type") == "collapse":
            output += f"## Collapse Details\n\n"
            content = trace_data.get("content", {})
            output += f"**Type:** {content.get('collapse_type', 'N/A')}\n"
            output += f"**Reason:** {content.get('collapse_reason', 'N/A')}\n\n"
            
            # Add details if available
            if "details" in content:
                output += f"## Collapse Details\n\n"
                details = content["details"]
                for key, value in details.items():
                    output += f"**{key}:** {value}\n"
        
        elif trace_data.get("trace_type") == "shell":
            output += f"## Shell Details\n\n"
            content = trace_data.get("content", {})
            output += f"**Shell Pattern:** {content.get('shell_pattern', 'N/A')}\n\n"
            
            # Add content details
            shell_content = content.get("content", {})
            output += f"## Shell Content\n\n"
            for key, value in shell_content.items():
                output += f"**{key}:** {value}\n"
        
        return output
    
    else:  # text format (default)
        # Create text output
        output = "==== Diagnostic Trace ====\n\n"
        
        # Add trace info
        output += f"Trace ID: {trace_data.get('trace_id', 'N/A')}\n"
        output += f"Agent: {trace_data.get('agent_name', 'N/A')}\n"
        output += f"Type: {trace_data.get('trace_type', 'N/A')}\n"
        output += f"Time: {trace_data.get('timestamp', 'N/A')}\n\n"
        
        # Add shell patterns if available
        if "shell_patterns" in trace_data:
            output += f"Shell Patterns:\n"
            for pattern in trace_data["shell_patterns"]:
                output += f"- {pattern}\n"
            output += "\n"
        
        # Add content based on trace type
        content = trace_data.get("content", {})
        output += f"---- Content ----\n\n"
        
        # Format content recursively
        def format_dict(d, indent=0):
            result = ""
            for key, value in d.items():
                if isinstance(value, dict):
                    result += f"{'  ' * indent}{key}:\n"
                    result += format_dict(value, indent + 1)
                elif isinstance(value, list):
                    result += f"{'  ' * indent}{key}:\n"
                    for item in value:
                        if isinstance(item, dict):
                            result += format_dict(item, indent + 1)
                        else:
                            result += f"{'  ' * (indent + 1)}- {item}\n"
                else:
                    result += f"{'  ' * indent}{key}: {value}\n"
            return result
        
        output += format_dict(content)
        
        return output


def get_shell_pattern_description(pattern: ShellPattern) -> str:
    """
    Get description for a shell pattern.
    
    Args:
        pattern: Shell pattern
        
    Returns:
        Shell pattern description
    """
    descriptions = {
        ShellPattern.NULL_FEATURE: "Knowledge gaps as null attribution zones",
        ShellPattern.CIRCUIT_FRAGMENT: "Broken reasoning paths in attribution chains",
        ShellPattern.META_FAILURE: "Metacognitive attribution failures",
        ShellPattern.GHOST_FRAME: "Residual agent identity markers",
        ShellPattern.ECHO_ATTRIBUTION: "Causal chain backpropagation",
        ShellPattern.ATTRIBUTION_REFLECT: "Multi-head contribution analysis",
        ShellPattern.INVERSE_CHAIN: "Attribution-output mismatch",
        ShellPattern.RECURSIVE_FRACTURE: "Circular attribution loops",
        ShellPattern.ETHICAL_INVERSION: "Value polarity reversals",
        ShellPattern.RESIDUAL_ALIGNMENT_DRIFT: "Direction of belief evolution",
    }
    
    return descriptions.get(pattern, "Unknown shell pattern")
