"""
AttributionTracer - Decision Provenance and Causal Tracing Framework

This module implements the attribution tracing architecture that enables
transparent decision provenance for all agents in the AGI-HEDGE-FUND system.

Key capabilities:
- Multi-level attribution across reasoning chains
- Causal tracing from decision back to evidence
- Confidence weighting of attribution factors
- Value-weighted attribution alignment
- Attribution visualization for interpretability

Internal Note: The attribution tracer encodes the ECHO-ATTRIBUTION and ATTRIBUTION-REFLECT
interpretability shells for causal path tracing and attribution transparency.
"""

import datetime
import uuid
import math
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict

from pydantic import BaseModel, Field


class AttributionEntry(BaseModel):
    """Single attribution entry linking a decision to a cause."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(...)  # Source ID (e.g., memory ID, evidence ID)
    source_type: str = Field(...)  # Type of source (e.g., "memory", "evidence", "reasoning")
    target: str = Field(...)  # Target ID (e.g., decision ID, reasoning step)
    weight: float = Field(default=1.0)  # Attribution weight (0-1)
    confidence: float = Field(default=1.0)  # Confidence in attribution (0-1)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    description: Optional[str] = Field(default=None)  # Optional attribution description
    value_alignment: Optional[float] = Field(default=None)  # Alignment with agent values (0-1)


class AttributionChain(BaseModel):
    """Chain of attribution entries forming a causal path."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entries: List[AttributionEntry] = Field(default_factory=list)
    start_point: str = Field(...)  # ID of chain origin
    end_point: str = Field(...)  # ID of chain destination
    total_weight: float = Field(default=1.0)  # Product of weights along chain
    confidence: float = Field(default=1.0)  # Overall chain confidence
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)


class AttributionGraph(BaseModel):
    """Complete attribution graph for a decision."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str = Field(...)  # ID of the decision being attributed
    chains: List[AttributionChain] = Field(default_factory=list)
    sources: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # Source metadata
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)
    
    def add_chain(self, chain: AttributionChain) -> None:
        """Add attribution chain to graph."""
        self.chains.append(chain)
    
    def add_source(self, source_id: str, metadata: Dict[str, Any]) -> None:
        """Add source metadata to graph."""
        self.sources[source_id] = metadata
    
    def calculate_source_contributions(self) -> Dict[str, float]:
        """Calculate normalized contribution of each source to decision."""
        # Initialize contributions
        contributions = defaultdict(float)
        
        # Sum weights from all chains
        for chain in self.chains:
            for entry in chain.entries:
                # Add contribution weighted by chain confidence
                contributions[entry.source] += entry.weight * chain.confidence
        
        # Normalize contributions
        total = sum(contributions.values())
        if total > 0:
            for source in contributions:
                contributions[source] /= total
        
        return dict(contributions)


class AttributionTracer:
    """
    Attribution tracing engine for causal decision provenance.
    
    Enables:
    - Tracing the causal path from decisions back to evidence
    - Weighting attribution factors by confidence and relevance
    - Aligning attribution with agent value system
    - Visualizing attribution patterns for interpretability
    """
    
    def __init__(self):
        """Initialize attribution tracer."""
        self.attribution_history: Dict[str, AttributionGraph] = {}
        self.trace_registry: Dict[str, Dict[str, Any]] = {}
        self.value_weights: Dict[str, float] = {}
    
    def trace_attribution(self, signal: Dict[str, Any], agent_state: Dict[str, Any],
                       reasoning_depth: int = 3) -> Dict[str, Any]:
        """
        Trace attribution for a decision signal.
        
        Args:
            signal: Decision signal
            agent_state: Agent's current state
            reasoning_depth: Depth of attribution tracing
            
        Returns:
            Attribution trace results
        """
        # Generate decision ID if not present
        decision_id = signal.get("signal_id", str(uuid.uuid4()))
        
        # Create attribution graph
        attribution_graph = AttributionGraph(
            decision_id=decision_id,
        )
        
        # Extract signal components for attribution
        ticker = signal.get("ticker", "")
        action = signal.get("action", "")
        confidence = signal.get("confidence", 0.5)
        reasoning = signal.get("reasoning", "")
        intent = signal.get("intent", "")
        value_basis = signal.get("value_basis", "")
        
        # Extract evidence sources from agent state
        evidence_sources = self._extract_evidence_sources(agent_state, ticker, action)
        
        # Process reasoning to extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(reasoning)
        
        # Generate attribution chains
        chains = self._generate_attribution_chains(
            decision_id=decision_id,
            evidence_sources=evidence_sources,
            reasoning_steps=reasoning_steps,
            intent=intent,
            value_basis=value_basis,
            confidence=confidence,
            reasoning_depth=reasoning_depth
        )
        
        # Add chains to graph
        for chain in chains:
            attribution_graph.add_chain(chain)
        
        # Add source metadata
        for source_id, metadata in evidence_sources.items():
            attribution_graph.add_source(source_id, metadata)
        
        # Calculate source contributions
        source_contributions = attribution_graph.calculate_source_contributions()
        
        # Store in history
        # Store in history
        self.attribution_history[decision_id] = attribution_graph
        
        # Prepare result
        trace_id = str(uuid.uuid4())
        
        # Store trace in registry
        self.trace_registry[trace_id] = {
            "attribution_graph": attribution_graph,
            "decision_id": decision_id,
            "timestamp": datetime.datetime.now(),
        }
        
        # Create attribution trace output
        attribution_trace = {
            "trace_id": trace_id,
            "decision_id": decision_id,
            "attribution_map": source_contributions,
            "confidence": confidence,
            "top_factors": self._get_top_attribution_factors(source_contributions, 5),
            "value_alignment": self._calculate_value_alignment(value_basis, source_contributions),
            "reasoning_depth": reasoning_depth,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        
        return attribution_trace
    
    def _extract_evidence_sources(self, agent_state: Dict[str, Any], 
                               ticker: str, action: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract evidence sources from agent state.
        
        Args:
            agent_state: Agent's current state
            ticker: Stock ticker
            action: Decision action
            
        Returns:
            Dictionary of evidence sources
        """
        evidence_sources = {}
        
        # Extract from belief state
        belief_state = agent_state.get("belief_state", {})
        if ticker in belief_state:
            source_id = f"belief:{ticker}"
            evidence_sources[source_id] = {
                "type": "belief",
                "ticker": ticker,
                "value": belief_state[ticker],
                "description": f"Belief about {ticker}",
            }
        
        # Extract from working memory
        working_memory = agent_state.get("working_memory", {})
        
        # Check for ticker-specific data in working memory
        if ticker in working_memory:
            source_id = f"working_memory:{ticker}"
            evidence_sources[source_id] = {
                "type": "working_memory",
                "ticker": ticker,
                "data": working_memory[ticker],
                "description": f"Current analysis of {ticker}",
            }
        
        # Extract from performance trace if action is based on past performance
        performance_trace = agent_state.get("performance_trace", {})
        if ticker in performance_trace:
            source_id = f"performance:{ticker}"
            evidence_sources[source_id] = {
                "type": "performance",
                "ticker": ticker,
                "performance": performance_trace[ticker],
                "description": f"Performance history of {ticker}",
            }
        
        # Extract from decision history
        decision_history = agent_state.get("decision_history", [])
        for i, decision in enumerate(decision_history):
            if decision.get("ticker") == ticker and decision.get("action") == action:
                source_id = f"past_decision:{i}:{ticker}"
                evidence_sources[source_id] = {
                    "type": "past_decision",
                    "ticker": ticker,
                    "action": action,
                    "decision": decision,
                    "description": f"Past {action} decision for {ticker}",
                }
        
        return evidence_sources
    
    def _extract_reasoning_steps(self, reasoning: str) -> List[Dict[str, Any]]:
        """
        Extract reasoning steps from reasoning string.
        
        Args:
            reasoning: Reasoning string
            
        Returns:
            List of reasoning steps
        """
        # Simple implementation: split by periods or line breaks
        sentences = [s.strip() for s in reasoning.replace('\n', '. ').split('.') if s.strip()]
        
        reasoning_steps = []
        for i, sentence in enumerate(sentences):
            step_id = f"step:{i}"
            reasoning_steps.append({
                "id": step_id,
                "text": sentence,
                "position": i,
                "type": "reasoning_step",
            })
        
        return reasoning_steps
    
    def _generate_attribution_chains(self, decision_id: str, evidence_sources: Dict[str, Dict[str, Any]],
                                  reasoning_steps: List[Dict[str, Any]], intent: str, value_basis: str,
                                  confidence: float, reasoning_depth: int) -> List[AttributionChain]:
        """
        Generate attribution chains linking decision to evidence.
        
        Args:
            decision_id: Decision ID
            evidence_sources: Evidence sources
            reasoning_steps: Reasoning steps
            intent: Decision intent
            value_basis: Value basis for decision
            confidence: Decision confidence
            reasoning_depth: Depth of attribution tracing
            
        Returns:
            List of attribution chains
        """
        attribution_chains = []
        
        # Define end point (the decision itself)
        end_point = decision_id
        
        # Case 1: Direct evidence -> decision chains
        for source_id, source_data in evidence_sources.items():
            # Create entry linking evidence directly to decision
            entry = AttributionEntry(
                source=source_id,
                source_type=source_data.get("type", "evidence"),
                target=decision_id,
                weight=self._calculate_evidence_weight(source_data, confidence),
                confidence=confidence,
                description=f"Direct influence of {source_data.get('description', source_id)} on decision",
            )
            
            # Create chain
            chain = AttributionChain(
                entries=[entry],
                start_point=source_id,
                end_point=end_point,
                total_weight=entry.weight,
                confidence=entry.confidence,
            )
            
            attribution_chains.append(chain)
        
        # Case 2: Evidence -> reasoning -> decision chains
        if reasoning_steps:
            # For each evidence source
            for source_id, source_data in evidence_sources.items():
                # For relevant reasoning steps (limited by depth)
                for step in reasoning_steps[:reasoning_depth]:
                    # Create entry linking evidence to reasoning step
                    step_entry = AttributionEntry(
                        source=source_id,
                        source_type=source_data.get("type", "evidence"),
                        target=step["id"],
                        weight=self._calculate_step_relevance(source_data, step),
                        confidence=confidence * 0.9,  # Slightly lower confidence for indirect paths
                        description=f"Influence of {source_data.get('description', source_id)} on reasoning step",
                    )
                    
                    # Create entry linking reasoning step to decision
                    decision_entry = AttributionEntry(
                        source=step["id"],
                        source_type="reasoning_step",
                        target=decision_id,
                        weight=self._calculate_step_importance(step, len(reasoning_steps)),
                        confidence=confidence,
                        description=f"Influence of reasoning step on decision",
                    )
                    
                    # Create chain
                    chain = AttributionChain(
                        entries=[step_entry, decision_entry],
                        start_point=source_id,
                        end_point=end_point,
                        total_weight=step_entry.weight * decision_entry.weight,
                        confidence=min(step_entry.confidence, decision_entry.confidence),
                    )
                    
                    attribution_chains.append(chain)
        
        # Case 3: Intent/value -> decision chains
        if intent:
            intent_id = f"intent:{intent[:20]}"
            intent_entry = AttributionEntry(
                source=intent_id,
                source_type="intent",
                target=decision_id,
                weight=0.8,  # High weight for intent
                confidence=confidence,
                description=f"Influence of stated intent on decision",
            )
            
            intent_chain = AttributionChain(
                entries=[intent_entry],
                start_point=intent_id,
                end_point=end_point,
                total_weight=intent_entry.weight,
                confidence=intent_entry.confidence,
            )
            
            attribution_chains.append(intent_chain)
        
        if value_basis:
            value_id = f"value:{value_basis[:20]}"
            value_entry = AttributionEntry(
                source=value_id,
                source_type="value",
                target=decision_id,
                weight=0.9,  # Very high weight for value basis
                confidence=confidence,
                description=f"Influence of value basis on decision",
                value_alignment=1.0,  # Perfect alignment with its own value
            )
            
            value_chain = AttributionChain(
                entries=[value_entry],
                start_point=value_id,
                end_point=end_point,
                total_weight=value_entry.weight,
                confidence=value_entry.confidence,
            )
            
            attribution_chains.append(value_chain)
        
        return attribution_chains
    
    def _calculate_evidence_weight(self, evidence: Dict[str, Any], base_confidence: float) -> float:
        """
        Calculate weight of evidence.
        
        Args:
            evidence: Evidence data
            base_confidence: Base confidence level
            
        Returns:
            Evidence weight
        """
        # Default weight
        weight = 0.5
        
        # Adjust based on evidence type
        evidence_type = evidence.get("type", "")
        
        if evidence_type == "belief":
            # Weight based on belief strength (0.5-1.0)
            belief_value = evidence.get("value", 0.5)
            weight = 0.5 + (abs(belief_value - 0.5) * 0.5)
        
        elif evidence_type == "working_memory":
            # Working memory has high weight
            weight = 0.8
        
        elif evidence_type == "performance":
            # Performance data moderately important
            weight = 0.7
        
        elif evidence_type == "past_decision":
            # Past decisions less important
            weight = 0.6
        
        # Scale by confidence
        weight *= base_confidence
        
        return min(1.0, weight)
    
    def _calculate_step_relevance(self, evidence: Dict[str, Any], step: Dict[str, Any]) -> float:
        """
        Calculate relevance of evidence to reasoning step.
        
        Args:
            evidence: Evidence data
            step: Reasoning step
            
        Returns:
            Relevance weight
        """
        # Basic implementation using text overlap
        evidence_desc = evidence.get("description", "")
        step_text = step.get("text", "")
        
        # Check for ticker mention
        ticker = evidence.get("ticker", "")
        if ticker and ticker in step_text:
            return 0.8
        
        # Check for word overlap
        evidence_words = set(evidence_desc.lower().split())
        step_words = set(step_text.lower().split())
        
        overlap = len(evidence_words.intersection(step_words))
        total_words = len(evidence_words.union(step_words))
        
        if total_words > 0:
            overlap_ratio = overlap / total_words
            return min(1.0, 0.5 + overlap_ratio)
        
        return 0.5
    
    def _calculate_step_importance(self, step: Dict[str, Any], total_steps: int) -> float:
        """
        Calculate importance of reasoning step.
        
        Args:
            step: Reasoning step
            total_steps: Total number of steps
            
        Returns:
            Importance weight
        """
        # Position-based importance (later steps slightly more important)
        position = step.get("position", 0)
        position_weight = 0.5 + (position / (2 * total_steps)) if total_steps > 0 else 0.5
        
        # Length-based importance (longer steps slightly more important)
        text = step.get("text", "")
        length = len(text)
        length_weight = min(1.0, 0.5 + (length / 200))  # Cap at 1.0
        
        # Combine weights
        return (position_weight * 0.7) + (length_weight * 0.3)
    
    def _get_top_attribution_factors(self, source_contributions: Dict[str, float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get top attribution factors.
        
        Args:
            source_contributions: Source contribution dictionary
            limit: Maximum number of factors to return
            
        Returns:
            List of top attribution factors
        """
        # Sort contributions by weight (descending)
        sorted_contributions = sorted(
            source_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Take top 'limit' contributions
        top_factors = []
        for source, weight in sorted_contributions[:limit]:
            # Parse source type from ID
            source_type = source.split(":", 1)[0] if ":" in source else "unknown"
            
            top_factors.append({
                "source": source,
                "type": source_type,
                "weight": weight,
            })
        
        return top_factors
    
    def _calculate_value_alignment(self, value_basis: str, source_contributions: Dict[str, float]) -> float:
        """
        Calculate value alignment score.
        
        Args:
            value_basis: Value basis string
            source_contributions: Source contribution dictionary
            
        Returns:
            Value alignment score
        """
        # Simple implementation: check if value sources have high contribution
        value_alignment = 0.5  # Default neutral alignment
        
        # Find value-based sources
        value_sources = [source for source in source_contributions if source.startswith("value:")]
        
        if value_sources:
            # Calculate contribution of value sources
            value_contribution = sum(source_contributions[source] for source in value_sources)
            
            # Value alignment increases with value contribution
            value_alignment = 0.5 + (value_contribution * 0.5)
        
        return min(1.0, value_alignment)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get attribution trace by ID.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Attribution trace or None if not found
        """
        if trace_id not in self.trace_registry:
            return None
        
        trace_data = self.trace_registry[trace_id]
        attribution_graph = trace_data.get("attribution_graph")
        
        if not attribution_graph:
            return None
        
        # Calculate source contributions
        source_contributions = attribution_graph.calculate_source_contributions()
        
        # Create attribution trace output
        attribution_trace = {
            "trace_id": trace_id,
            "decision_id": attribution_graph.decision_id,
            "attribution_map": source_contributions,
            "top_factors": self._get_top_attribution_factors(source_contributions, 5),
            "chains": len(attribution_graph.chains),
            "sources": len(attribution_graph.sources),
            "timestamp": trace_data.get("timestamp", datetime.datetime.now()).isoformat(),
        }
        
        return attribution_trace
    
    def get_decision_traces(self, decision_id: str) -> List[str]:
        """
        Get trace IDs for a decision.
        
        Args:
            decision_id: Decision ID
            
        Returns:
            List of trace IDs
        """
        return [trace_id for trace_id, trace_data in self.trace_registry.items()
              if trace_data.get("decision_id") == decision_id]
    
    def visualize_attribution(self, trace_id: str) -> Dict[str, Any]:
        """
        Generate attribution visualization data.
        
        Args:
            trace_id: Trace ID
            
        Returns:
            Visualization data
        """
        if trace_id not in self.trace_registry:
            return {"error": "Trace not found"}
        
        trace_data = self.trace_registry[trace_id]
        attribution_graph = trace_data.get("attribution_graph")
        
        if not attribution_graph:
            return {"error": "Attribution graph not found"}
        
        # Create nodes and links for visualization
        nodes = []
        links = []
        
        # Add decision node
        decision_id = attribution_graph.decision_id
        nodes.append({
            "id": decision_id,
            "type": "decision",
            "label": "Decision",
            "size": 15,
        })
        
        # Process all chains
        for chain_idx, chain in enumerate(attribution_graph.chains):
            # Add source node if not already added
            source_id = chain.start_point
            if not any(node["id"] == source_id for node in nodes):
                # Determine source type
                source_type = "unknown"
                if source_id.startswith("belief:"):
                    source_type = "belief"
                elif source_id.startswith("working_memory:"):
                    source_type = "working_memory"
                elif source_id.startswith("performance:"):
                    source_type = "performance"
                elif source_id.startswith("past_decision:"):
                    source_type = "past_decision"
                elif source_id.startswith("intent:"):
                    source_type = "intent"
                elif source_id.startswith("value:"):
                    source_type = "value"
                
                # Add source node
                nodes.append({
                    "id": source_id,
                    "type": source_type,
                    "label": source_id.split(":", 1)[1] if ":" in source_id else source_id,
                    "size": 10,
                })
            
            # Process chain entries
            prev_node_id = None
            for entry_idx, entry in enumerate(chain.entries):
                source_node_id = entry.source
                target_node_id = entry.target
                
                # Add intermediate nodes if not already added
                if entry.source_type == "reasoning_step" and not any(node["id"] == source_node_id for node in nodes):
                    nodes.append({
                        "id": source_node_id,
                        "type": "reasoning_step",
                        "label": f"Step {source_node_id.split(':', 1)[1] if ':' in source_node_id else source_node_id}",
                        "size": 8,
                    })
                
                # Add link
                links.append({
                    "source": source_node_id,
                    "target": target_node_id,
                    "value": entry.weight,
                    "confidence": entry.confidence,
                    "label": entry.description if entry.description else f"Weight: {entry.weight:.2f}",
                })
                
                prev_node_id = target_node_id
        
        # Create visualization data
        visualization = {
            "nodes": nodes,
            "links": links,
            "trace_id": trace_id,
            "decision_id": decision_id,
        }
        
        return visualization
    
    def set_value_weights(self, value_weights: Dict[str, float]) -> None:
        """
        Set weights for different values.
        
        Args:
            value_weights: Dictionary mapping value names to weights
        """
        self.value_weights = value_weights.copy()
    
    def clear_history(self, before_timestamp: Optional[datetime.datetime] = None) -> int:
        """
        Clear attribution history.
        
        Args:
            before_timestamp: Optional timestamp to clear history before
            
        Returns:
            Number of entries cleared
        """
        if before_timestamp is None:
            # Clear all history
            count = len(self.attribution_history)
            self.attribution_history = {}
            self.trace_registry = {}
            return count
        
        # Clear history before timestamp
        to_remove_history = []
        to_remove_registry = []
        
        for decision_id, graph in self.attribution_history.items():
            if graph.timestamp < before_timestamp:
                to_remove_history.append(decision_id)
        
        for trace_id, trace_data in self.trace_registry.items():
            if trace_data.get("timestamp", datetime.datetime.now()) < before_timestamp:
                to_remove_registry.append(trace_id)
        
        # Remove from history
        for decision_id in to_remove_history:
            del self.attribution_history[decision_id]
        
        # Remove from registry
        for trace_id in to_remove_registry:
            del self.trace_registry[trace_id]
        
        return len(to_remove_history) + len(to_remove_registry)
