# AGI-HEDGE-FUND Architecture Overview

<div align="center">
   <img src="assets/images/architecture_diagram.png" alt="AGI-HEDGE-FUND Architecture" width="800"/>
   <p><i>Multi-agent recursive market cognition framework</i></p>
</div>

## Recursive Cognitive Architecture

AGI-HEDGE-FUND implements a multi-layer recursive cognitive architecture that allows for deep reasoning, interpretable decision making, and emergent market understanding. The system operates through nested cognitive loops that implement a transparent and traceable decision framework.

### Architectural Layers

1. **Agent Cognitive Layer**
   - Philosophical agent implementations with distinct investment approaches
   - Each agent maintains its own memory shell, belief state, and reasoning capabilities
   - Attribution tracing for decision provenance
   - Shell-based diagnostic patterns for interpretability

2. **Multi-Agent Arbitration Layer**
   - Meta-agent for recursive consensus formation
   - Attribution-weighted position sizing
   - Conflict detection and resolution through value alignment
   - Emergent portfolio strategy through agent weighting

3. **Model Orchestration Layer**
   - Provider-agnostic LLM interface
   - Dynamic routing based on capabilities
   - Fallback mechanisms for reliability
   - Output parsing and normalization

4. **Market Interface Layer**
   - Data source abstractions
   - Backtest environment
   - Live market connection
   - Portfolio management

5. **Diagnostic & Interpretability Layer**
   - Tracing utilities for attribution visualization
   - Shell pattern detection for failure modes
   - Consensus graph generation
   - Agent conflict mapping

## Agent Architecture

Agents in AGI-HEDGE-FUND implement a recursive cognitive architecture with the following components:

### Memory Shell

The memory shell provides persistent state across market cycles with configurable decay rates. It includes:

- **Working Memory**: Active processing and temporary storage
- **Episodic Memory**: Experiences and past decisions with emotional valence
- **Semantic Memory**: Conceptual knowledge with certainty levels

Memory traces can be accessed through attribution pathways, enabling transparent decision tracing.

### Reasoning Graph

The reasoning graph implements a multi-step reasoning process using LangGraph with:

- Recursive reasoning loops with configurable depth
- Attribution tracing for causal relationships
- Collapse detection for reasoning failures
- Value-weighted decision making

Reasoning graphs can be visualized and inspected through the `--show-trace` flag.

### Belief State

Agents maintain an evolving belief state that:

- Tracks confidence in various market hypotheses
- Updates based on market feedback
- Drifts over time with configurable decay
- Influences decision weighting

Belief drift can be monitored through `.p/` command equivalents like `drift.observe{vector, bias}`.

## Portfolio Meta-Agent

The portfolio meta-agent serves as a recursive arbitration layer that:

1. Collects signals from all philosophical agents
2. Forms consensus through attribution-weighted aggregation
3. Resolves conflicts based on agent performance and reasoning quality
4. Sizes positions according to confidence and attribution
5. Maintains its own memory and learning from market feedback

### Consensus Formation Process

<div align="center">
   <img src="assets/images/consensus_process.png" alt="Consensus Formation Process" width="600"/>
</div>

The consensus formation follows a recursive process:

1. **Signal Generation**: Each agent processes market data through its philosophical lens
2. **Initial Consensus**: Non-conflicting signals form preliminary consensus
3. **Conflict Resolution**: Conflicting signals are resolved through attribution weighting
4. **Position Sizing**: Confidence and attribution determine position sizes
5. **Meta Reflection**: The meta-agent reflects on its decision process 
6. **Agent Weighting**: Agent weights are adjusted based on performance

### Agent Weighting

The meta-agent dynamically adjusts agent weights based on performance, consistency, and value alignment. This creates an emergent portfolio strategy that evolves over time through recursive performance evaluation.

The weighting formula combines:
- Historical returns attribution
- Win rate
- Consistency score
- Confidence calibration

This creates a dynamic, self-optimizing meta-strategy that adapts to changing market conditions while maintaining interpretable decision paths.

## Recursive Tracing Architecture

A key feature of AGI-HEDGE-FUND is its recursive tracing architecture that enables complete visibility into decision processes. This is implemented through:

### Attribution Tracing

Attribution tracing connects decisions to their causal origins through a multi-layered graph:

1. **Source Attribution**: Linking decisions to specific evidence or beliefs
2. **Reasoning Attribution**: Tracking steps in the reasoning process
3. **Value Attribution**: Connecting decisions to philosophical values
4. **Temporal Attribution**: Linking decisions across time

Attribution chains can be visualized with the `--attribution-report` flag.

### Shell Pattern Detection

The system implements interpretability shells inspired by circuit interpretability research. These detect specific reasoning patterns and potential failure modes:

| Shell Pattern | Description | Detection Mechanism |
|---------------|-------------|---------------------|
| NULL_FEATURE  | Knowledge gaps as null attribution zones | Confidence drops below threshold, belief gaps |
| CIRCUIT_FRAGMENT | Broken reasoning paths in attribution chains | Discontinuities in reasoning steps |
| META_FAILURE | Metacognitive attribution failures | Recursive errors beyond threshold depth |
| GHOST_FRAME | Residual agent identity markers | Identity persistence above threshold |
| ECHO_ATTRIBUTION | Causal chain backpropagation | Attribution path length beyond threshold |
| RECURSIVE_FRACTURE | Circular attribution loops | Repeating patterns in reasoning steps |
| ETHICAL_INVERSION | Value polarity reversals | Conflicting value attributions |
| RESIDUAL_ALIGNMENT_DRIFT | Direction of belief evolution | Belief drift magnitude above threshold |

Shell patterns can be visualized with the `--shell-failure-map` flag.

## Symbolic Command Interface

The system implements an internal symbolic command interface for agent communication and diagnostic access. These commands are inspired by circuit interpretability research and enable deeper introspection:

### Core Commands

- `.p/reflect.trace{agent, depth}`: Trace agent's self-reflection on decision making
- `.p/fork.signal{source}`: Fork a new signal branch from specified source
- `.p/collapse.detect{threshold, reason}`: Detect potential decision collapse
- `.p/attribute.weight{justification}`: Compute attribution weight for justification
- `.p/drift.observe{vector, bias}`: Observe and record belief drift

These commands are used internally by the system but can be exposed through diagnostic flags for advanced users.

## Model Router Architecture

The model router provides a unified interface for multiple language model providers:

<div align="center">
   <img src="assets/images/model_router.png" alt="Model Router Architecture" width="600"/>
</div>

### Provider Integration

The system supports multiple LLM providers:
- **OpenAI**: GPT-4, GPT-3.5-Turbo
- **Anthropic**: Claude 3 Opus, Sonnet, Haiku
- **Groq**: Llama, Mixtral
- **Ollama**: Local models
- **DeepSeek**: DeepSeek models

Each provider is integrated through a standard interface with fallback chains for reliability.

### Model Selection Logic

Models are selected based on:
1. Required capabilities (reasoning, finance domain knowledge, etc.)
2. Performance characteristics
3. Cost considerations
4. Availability

The system can automatically fall back to alternative providers if the primary provider is unavailable or fails.

## Memory Architecture

The memory architecture enables temporal persistence across market cycles:

<div align="center">
   <img src="assets/images/memory_architecture.png" alt="Memory Architecture" width="600"/>
</div>

### Memory Components

1. **Working Memory**: Short-term active processing with limited capacity
2. **Episodic Memory**: Experience-based memory with emotional valence and decay
3. **Semantic Memory**: Conceptual knowledge with certainty levels
4. **Temporal Sequence**: Ordered episodic memory for temporal reasoning

### Memory Operations

- **Add Experience**: Record market experiences with attribution
- **Query Memories**: Retrieve relevant memories based on context
- **Apply Decay**: Simulate memory decay over time
- **Consolidate Memories**: Convert episodic to semantic memories

## Extending the Framework

The system is designed for extensibility at multiple levels:

### Custom Agents

New philosophical agents can be added by extending the BaseAgent class:

```python
from agi_hedge_fund.agents import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self, reasoning_depth=3, memory_decay=0.2):
        super().__init__(
            name="Custom",
            philosophy="My unique investment approach",
            reasoning_depth=reasoning_depth,
            memory_decay=memory_decay
        )
        
    def process_market_data(self, data):
        # Custom market data processing
        pass
        
    def generate_signals(self, processed_data):
        # Custom signal generation
        pass
```

### Custom LLM Providers

New LLM providers can be added by extending the ModelProvider class:

```python
from agi_hedge_fund.llm.router import ModelProvider

class CustomProvider(ModelProvider):
    def __init__(self, api_key=None):
        # Initialize provider
        pass
        
    def generate(self, prompt, **kwargs):
        # Generate text using custom provider
        pass
        
    def get_available_models(self):
        # Return list of available models
        pass
        
    def get_model_capabilities(self, model_name):
        # Return capabilities of specific model
        pass
```

### Custom Shell Patterns

New interpretability shell patterns can be added by extending the ShellPattern enum:

```python
from agi_hedge_fund.utils.diagnostics import ShellPattern

# Add new shell pattern
ShellPattern.CUSTOM_PATTERN = "v999 CUSTOM-PATTERN"

# Configure shell pattern detection
shell_diagnostics.shell_patterns[ShellPattern.CUSTOM_PATTERN] = {
    "pattern": r"custom.*pattern|unique.*signature",
    "threshold": 0.5,
}
```

## Recursive Arbitration Mechanisms

The system implements several mechanisms for recursive arbitration:

### Consensus Formation

1. **Signal Collection**: Gather signals from all agents
2. **Signal Grouping**: Group signals by ticker and action
3. **Confidence Weighting**: Weight signals by agent confidence and performance
4. **Conflict Detection**: Identify conflicting signals
5. **Conflict Resolution**: Resolve conflicts through attribution weighting
6. **Consensus Decision**: Generate final consensus decisions

### Adaptive Weighting

Agents are weighted based on:
1. **Historical Performance**: Track record of successful decisions
2. **Consistency**: Alignment between reasoning and outcomes
3. **Calibration**: Accuracy of confidence estimates
4. **Value Alignment**: Consistency with portfolio philosophy

Weights evolve over time through recursive performance evaluation.

### Position Sizing

Position sizes are determined by:
1. **Signal Confidence**: Higher confidence = larger position
2. **Agent Attribution**: Weighted by agent performance
3. **Risk Budget**: Overall risk allocation constraints
4. **Min/Max Position Size**: Configurable position size limits

## Diagnostic Tools

The system includes diagnostic tools for interpretability:

### Tracing Tools

- **Signal Tracing**: Track signal flow through the system
- **Reasoning Tracing**: Visualize reasoning steps
- **Collapse Detection**: Identify reasoning failures
- **Shell Pattern Detection**: Detect specific interpretability patterns

### Visualization Tools

- **Consensus Graph**: Visualize multi-agent consensus formation
- **Conflict Map**: Map conflicts between agents
- **Attribution Report**: Visualize decision attribution
- **Shell Failure Map**: Map shell pattern failures

## Conclusion

The AGI-HEDGE-FUND architecture provides a recursive cognitive framework for multi-agent market understanding with complete transparency and interpretability. By combining philosophical agent archetypes, recursive reasoning, attribution tracing, and emergent meta-strategy, it enables a new approach to financial decision making that is both effective and explainable.

The system's design principles—recursion, attribution, interpretability, and emergence—create a platform that goes beyond traditional algorithmic trading to implement a true cognitive approach to market understanding.

---
