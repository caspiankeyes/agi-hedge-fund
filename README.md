<div align="center">

  <h1>AGI-HEDGE-FUND</h1>
  <p><i>Multi-agent recursive market cognition framework</i></p>

  [![GitHub Stars](https://img.shields.io/github/stars/agi-hedge-fund/agi-hedge-fund?style=flat-square)](https://github.com/agi-hedge-fund/agi-hedge-fund/stargazers)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
  [![GitHub Issues](https://img.shields.io/github/issues/agi-hedge-fund/agi-hedge-fund?style=flat-square)](https://github.com/agi-hedge-fund/agi-hedge-fund/issues)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Cognition Depth](https://img.shields.io/badge/cognition%20depth-recursive-purple.svg)](docs/ARCHITECTURE.md)
</div>

> **Internal Document: Anthropic Leadership & Executive Team**  
> **Classification: Technical Reference Documentation**  
> **Version: 0.9.3-alpha**  
> **Last Updated: 2025-04-17**
>
> 
## **`Overview`**

**AGI-HEDGE-FUND** is an experimental open-source framework that approaches financial markets as complex adaptive systems requiring recursive cognitive architectures for interpretation. Unlike traditional algorithmic trading, AGI-HEDGE-FUND implements a multi-agent system where each agent embodies a distinct investment philosophy, enabling emergent market understanding through recursive arbitration and attribution-weighted consensus.

> *"Markets are efficient precisely to the extent that recursive cognition can penetrate their complexity."*

## **`Key Features`**

- **Philosophical Agent Lattice**: Specialized agents embodying distinct investment philosophies from value investing to disruptive innovation
- **Recursive Reasoning Architecture**: LangGraph-powered reasoning loops with transparent attribution paths
- **Model-Agnostic Cognition**: Support for OpenAI, Anthropic, Groq, Ollama, and DeepSeek models
- **Temporal Memory Shells**: Agents maintain persistent state across market cycles
- **Attribution-Weighted Decisions**: Every trade includes fully traceable decision provenance
- **Interpretability Scaffolding**: `--show-trace` flag reveals complete reasoning paths
- **Real-Time Market Integration**: Connect to Alpha Vantage, Polygon.io, and Yahoo Finance
- **Backtesting Framework**: Test agent performance against historical market data
- **Portfolio Meta-Agent**: Emergent consensus mechanism with adaptive drift correction

## 📊 Performance Visualization

### **`In Production`**

## **`Agent Architecture`**

AGI-HEDGE-FUND implements a lattice of cognitive agents, each embodying a distinct investment philosophy and decision framework:

| Agent | Philosophy | Cognitive Signature | Time Horizon |
|-------|------------|---------------------|-------------|
| Graham | Value Investing | Undervalued Asset Detection | Long-term |
| Wood | Disruptive Innovation | Exponential Growth Projection | Long-term |
| Dalio | Macroeconomic Analysis | Economic Machine Modeling | Medium-term |
| Ackman | Activist Investing | Position Conviction & Advocacy | Medium-term |
| Simons | Statistical Arbitrage | Pattern Detection & Exploitation | Short-term |
| Taleb | Anti-fragility | Black Swan Preparation | All horizons |
| Meta | Arbitration & Consensus | Recursive Integration | Adaptive |

Each agent processes market data through its unique cognitive lens, contributing signals to the portfolio meta-agent which recursively arbitrates and integrates perspectives.

## **`Recursive Cognition Flow`**

<div align="center">
  
### **`In Production`**

The system operates through nested cognitive loops that implement a recursive market interpretation framework:

1. **Market Signal Perception**: Raw data ingestion and normalization
2. **Agent-Specific Processing**: Philosophy-aligned interpretation
3. **Multi-Agent Deliberation**: Signal exchange and position debate
4. **Recursive Arbitration**: Meta-agent integration and resolution
5. **Position Formulation**: Final decision synthesis with attribution
6. **Temporal Reflection**: Performance evaluation and belief updating

## **`Installation`**

```bash
# Clone the repository
git clone https://github.com/agi-hedge-fund/agi-hedge-fund.git
cd agi-hedge-fund

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## **`Quick Start`**

```python
from agi_hedge_fund import PortfolioManager
from agi_hedge_fund.agents import GrahamAgent, WoodAgent, DalioAgent
from agi_hedge_fund.market import MarketEnvironment

# Initialize market environment
market = MarketEnvironment(data_source="yahoo", tickers=["AAPL", "MSFT", "GOOGL", "AMZN"])

# Create agents with different cognitive depths
agents = [
    GrahamAgent(reasoning_depth=3),
    WoodAgent(reasoning_depth=4),
    DalioAgent(reasoning_depth=3)
]

# Initialize portfolio manager with recursive arbitration
portfolio = PortfolioManager(
    agents=agents,
    initial_capital=100000,
    arbitration_depth=2,
    show_trace=True
)

# Run simulation
results = portfolio.run_simulation(
    start_date="2020-01-01",
    end_date="2023-01-01",
    rebalance_frequency="weekly"
)

# Analyze results
portfolio.show_performance()
portfolio.generate_attribution_report()
portfolio.visualize_consensus_graph()
```

## **`Interpretability`**

AGI-HEDGE-FUND prioritizes transparent decision-making through recursive attribution tracing. Use the following flags to inspect agent cognition:

```bash
# Run with complete reasoning trace
python -m agi_hedge_fund.run --show-trace

# Visualize agent consensus formation
python -m agi_hedge_fund.run --consensus-graph

# Map conflicts in multi-agent deliberation
python -m agi_hedge_fund.run --agent-conflict-map

# Generate attribution report for all trades
python -m agi_hedge_fund.run --attribution-report
```

## **`Extending the Framework`**

The system is designed for extensibility at multiple levels:

### Creating Custom Agents

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
        # Implement custom market interpretation logic
        processed_data = self.cognitive_shell.process(data)
        return processed_data
        
    def generate_signals(self, processed_data):
        # Generate investment signals with attribution
        signals = self.reasoning_graph.run(
            input=processed_data,
            trace_depth=self.reasoning_depth
        )
        return self.attribute_signals(signals)
```

### Customizing the Arbitration Layer

```python
from agi_hedge_fund.cognition import ArbitrationMechanism

class CustomArbitration(ArbitrationMechanism):
    def __init__(self, weighting_strategy="confidence"):
        super().__init__(weighting_strategy=weighting_strategy)
        
    def resolve_conflicts(self, signals):
        # Implement custom conflict resolution logic
        resolution = self.recursive_integration(signals)
        return resolution
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [LangGraph](https://github.com/langchain-ai/langgraph) - Framework for building stateful, multi-actor applications with LLMs
- [Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT) - Autonomous GPT-4 experiment
- [LangChain](https://github.com/langchain-ai/langchain) - Building applications with LLMs
- [Fintech-LLM](https://github.com/AI4Finance-Foundation/FinGPT) - Financial language models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

## 📚 Citation

If you use AGI-HEDGE-FUND in your research, please cite:

```bibtex
@software{agi_hedge_fund2024,
  author = {{AGI-HEDGE-FUND Contributors}},
  title = {AGI-HEDGE-FUND: Multi-agent recursive market cognition framework},
  url = {https://github.com/agi-hedge-fund/agi-hedge-fund},
  year = {2024},
}
```

## 🌟 Acknowledgements

- The philosophical agents are inspired by the investment approaches of Benjamin Graham, Cathie Wood, Ray Dalio, Bill Ackman, Jim Simons, and Nassim Nicholas Taleb
- Recursive reasoning architecture influenced by work in multi-agent systems and interpretability research
- Market simulation components build upon open-source financial analysis libraries

---

<div align="center">
  <p>Built with ❤️ by the AGI-HEDGE-FUND team</p>
  <p><i>Recursion. Interpretation. Emergence.</i></p>
</div>
