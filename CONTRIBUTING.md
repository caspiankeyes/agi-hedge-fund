# Contributing to AGI Hedge Fund

Thank you for your interest in contributing to AGI-HEDGE-FUND! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Project Structure](#project-structure)
- [Contributing Code](#contributing-code)
- [Adding New Agents](#adding-new-agents)
- [Adding New LLM Providers](#adding-new-llm-providers)
- [Extending Diagnostic Tools](#extending-diagnostic-tools)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Core Development Principles](#core-development-principles)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork to your local machine
3. Set up the development environment
4. Make your changes
5. Submit a pull request

## Development Environment

To set up your development environment:

```bash
# Clone the repository
git clone https://github.com/your-username/agi-hedge-fund.git
cd agi-hedge-fund

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"
```

## Project Structure

Understanding the project structure is important for effective contributions:

```
agi-hedge-fund/
├── src/
│   ├── agents/                  # Agent implementations
│   │   ├── base.py              # Base agent architecture
│   │   ├── graham.py            # Value investor agent
│   │   ├── wood.py              # Innovation investor agent
│   │   └── ...                  # Other agent implementations
│   ├── cognition/               # Recursive reasoning framework
│   │   ├── graph.py             # LangGraph reasoning implementation
│   │   ├── memory.py            # Temporal memory shell
│   │   ├── attribution.py       # Decision attribution tracing
│   │   └── arbitration.py       # Consensus mechanisms
│   ├── market/                  # Market data interfaces
│   │   ├── sources/             # Data provider integrations
│   │   ├── environment.py       # Market simulation environment
│   │   └── backtesting.py       # Historical testing framework
│   ├── llm/                     # Language model integrations
│   │   ├── models/              # Model-specific implementations
│   │   ├── router.py            # Multi-model routing logic
│   │   └── prompts/             # Structured prompting templates
│   ├── utils/                   # Utility functions
│   │   ├── diagnostics/         # Interpretability tools
│   │   ├── visualization.py     # Performance visualization
│   │   └── metrics.py           # Performance metrics
│   ├── portfolio/               # Portfolio management
│   │   ├── manager.py           # Core portfolio manager
│   │   ├── allocation.py        # Position sizing logic
│   │   └── risk.py              # Risk management
│   └── main.py                  # Entry point
├── examples/                    # Example usage scripts
├── tests/                       # Test suite
├── docs/                        # Documentation
└── notebooks/                   # Jupyter notebooks
```

## Contributing Code

We follow a standard GitHub flow:

1. Create a new branch from `main` for your feature or bugfix
2. Make your changes
3. Add tests for your changes
4. Run the test suite to ensure all tests pass
5. Format your code with Black
6. Submit a pull request to `main`

### Coding Style

We follow these coding standards:

- Use [Black](https://github.com/psf/black) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) naming conventions
- Use type hints for function signatures
- Write docstrings in the Google style

To check and format your code:

```bash
# Format code with Black
black src tests examples

# Sort imports with isort
isort src tests examples

# Run type checking with mypy
mypy src
```

## Adding New Agents

To add a new philosophical agent:

1. Create a new file in `src/agents/` following existing agents as templates
2. Extend the `BaseAgent` class
3. Implement required methods: `process_market_data` and `generate_signals`
4. Add custom reasoning nodes to the agent's reasoning graph
5. Set appropriate memory decay and reasoning depth parameters
6. Add tests in `tests/agents/`

Example:

```python
from agi_hedge_fund.agents.base import BaseAgent, AgentSignal

class MyNewAgent(BaseAgent):
    def __init__(
        self,
        reasoning_depth: int = 3,
        memory_decay: float = 0.2,
        initial_capital: float = 100000.0,
        model_provider: str = "anthropic",
        model_name: str = "claude-3-sonnet-20240229",
        trace_enabled: bool = False,
    ):
        super().__init__(
            name="MyNew",
            philosophy="My unique investment philosophy",
            reasoning_depth=reasoning_depth,
            memory_decay=memory_decay,
            initial_capital=initial_capital,
            model_provider=model_provider,
            model_name=model_name,
            trace_enabled=trace_enabled,
        )
        
        # Configure reasoning graph
        self._configure_reasoning_graph()
    
    def _configure_reasoning_graph(self) -> None:
        """Configure the reasoning graph with custom nodes."""
        # Add custom reasoning nodes
        self.reasoning_graph.add_node(
            "my_custom_analysis",
            self._my_custom_analysis
        )
        
        # Configure reasoning flow
        self.reasoning_graph.set_entry_point("my_custom_analysis")
        
    def process_market_data(self, data):
        # Implement custom market data processing
        pass
        
    def generate_signals(self, processed_data):
        # Implement custom signal generation
        pass
        
    def _my_custom_analysis(self, state):
        # Implement custom reasoning node
        pass
```

## Adding New LLM Providers

To add a new LLM provider:

1. Extend the `ModelProvider` class in `src/llm/router.py`
2. Implement required methods
3. Update the `ModelRouter` to include your provider
4. Add tests in `tests/llm/`

Example:

```python
from agi_hedge_fund.llm.router import ModelProvider, ModelCapability

class MyCustomProvider(ModelProvider):
    """Custom model provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize custom provider.
        
        Args:
            api_key: API key (defaults to environment variable)
        """
        self.api_key = api_key or os.environ.get("MY_CUSTOM_API_KEY")
        
        # Define models and capabilities
        self.models = {
            "my-custom-model": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
            ],
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        # Implementation
        pass
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        return self.models.get(model_name, [])
```

## Extending Diagnostic Tools

To add new diagnostic capabilities:

1. Add new shell patterns in `src/utils/diagnostics.py`
2. Implement detection logic
3. Update visualization tools to support the new pattern
4. Add tests in `tests/utils/`

Example:

```python
from agi_hedge_fund.utils.diagnostics import ShellPattern

# Add new shell pattern
class MyCustomShellPattern(ShellPattern):
    CUSTOM_PATTERN = "v999 CUSTOM-PATTERN"

# Configure shell pattern detection
shell_diagnostics.shell_patterns[MyCustomShellPattern.CUSTOM_PATTERN] = {
    "pattern": r"custom.*pattern|unique.*signature",
    "custom_threshold": 0.5,
}

# Implement detection logic
def _detect_custom_pattern(self, trace_type: str, content: Dict[str, Any]) -> bool:
    content_str = json.dumps(content, ensure_ascii=False).lower()
    pattern = self.shell_patterns[MyCustomShellPattern.CUSTOM_PATTERN]["pattern"]
    
    # Check if pattern matches
    if re.search(pattern, content_str, re.IGNORECASE):
        # Add additional validation logic
        return custom_validation_logic(content)
    
    return False
```

## Documentation

Good documentation is crucial for the project. When contributing:

1. Update docstrings for any modified functions or classes
2. Update README.md if you're adding major features
3. Add examples for new features in the examples directory
4. Consider adding Jupyter notebooks for complex features

## Pull Request Process

1. Ensure your code follows our coding standards
2. Add tests for your changes
3. Update documentation as needed
4. Submit a pull request with a clear description of your changes
5. Address any feedback from reviewers

## Core Development Principles

When contributing to AGI-HEDGE-FUND, keep these core principles in mind:

1. **Transparency**: All agent decisions should be traceable and explainable
2. **Recursion**: Favor recursive approaches that enable deeper reasoning
3. **Attribution**: Maintain clear attribution chains for all decisions
4. **Interpretability**: Design for introspection and understanding
5. **Extensibility**: Make it easy to extend and customize the framework

By following these principles, you'll help maintain the project's coherence and quality.

Thank you for contributing to AGI-HEDGE-FUND!
