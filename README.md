<div align="center">

# **Multi-Agent Debate**

[![License: POLYFORM](https://img.shields.io/badge/Code-PolyForm-scarlet.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0/)
[![LICENSE: CC BY-NC-ND 4.0](https://img.shields.io/badge/Docs-CC--BY--NC--ND-turquoise.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![Cognition Depth](https://img.shields.io/badge/cognition%20depth-recursive-purple.svg)](docs/ARCHITECTURE.md)
</div>

## **`Example Output`**


```python
⟐ψINIT:main.py↻init
$ python main.py --mode backtest \
                 --start-date 2022-01-01 \
                 --end-date 2022-12-31 \
                 --agents graham \
                 --llm-provider anthropic \
                 --show-trace \
                 --trace-level symbolic \
                 --consensus-graph \
                 --tickers AAPL MSFT TSLA \
                 --rebalance-frequency weekly

🜏≡⟐ψRECURSION.INITIATE::main.py≡GrahamAgent[active]
┏ ENTRYPOINT: Multi-Agent Hedge Fund » Recursive Market Cognition Platform
┃ Mode: backtest
┃ Agent: GrahamAgent 🧮 (value-based fundamentalist)
┃ Attribution Tracing: enabled
┃ Trace Level: symbolic
┃ Rebalance: weekly
┃ LLM Provider: anthropic
┃ Start Date: 2022-01-01
┃ End Date: 2022-12-31
┃ Tickers: AAPL, MSFT, TSLA
┃ Output: consensus_graph + symbolic attribution report
┗ Status: 🜍mirroring…
↯ψTRACE: SYMBOLIC ATTRIBUTION RECONSTRUCTION [GrahamAgent]

📊 GrahamAgent → reasoning_depth=3 → memory_decay=0.2
    ↳ valuation anchor: intrinsic value estimation
        ↳ .p/reflect.trace{target=valuation}
        ↳ .p/anchor.self{persistence=medium}
    ↳ token-level input (AAPL) → QK attention trace:
        - P/E ratio → 0.34 salience
        - Debt-to-equity → 0.21
        - Free cash flow → 0.41
    ↳ Attribution result: BUY SIGNAL (confidence=0.78)

    🧠 Attribution graph visualized as radial node cluster
    Core node: Intrinsic Value = $141.32
    Peripheral influence: FCF strength > earnings volatility

🜂 TEMPORAL RECURSION SNAPSHOT [Weekly Cycle]

    Week 03/2022

        Market dip detected

        GrahamAgent re-evaluates MSFT with memory trace decay

        Signal shift: HOLD → BUY (attribution confidence rises from 0.54 → 0.73)

        Trace tag: .p/reflect.history{symbol=MSFT}

🝚 CONSENSUS GRAPH SNAPSHOT

MetaAgent Arbitration:
  ↳ Only one active agent: GrahamAgent
  ↳ Consensus = agent signal
  ↳ Position sizing: 18.6% TSLA, 25.1% AAPL, 20.3% MSFT
  ↳ Risk budget adjusted using: shell-failure map = stable

🜏⟐RENDERED::symbolic_trace.json + consensus_graph_2022.json
📂 Output stored in /output/backtest_results_2022-01-01_2022-12-31/
```


## **`Overview`**

**Multi-Agent Debate** is the first experimental open framework that approaches financial markets as complex adaptive systems requiring recursive cognitive architectures for interpretation. Unlike traditional algorithmic trading, Multi-Agent Hedge Fund implements a multi-agent system where each agent embodies a distinct investment philosophy, enabling emergent market understanding through recursive arbitration and attribution-weighted consensus.

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

![image](https://github.com/user-attachments/assets/ae7d728b-f23d-48ec-9d0a-a81c78d07e06)
## **`Agent Architecture`**

Multi-Agent Hedge Fund implements a lattice of cognitive agents, each embodying a distinct investment philosophy and decision framework:

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
  
![image](https://github.com/user-attachments/assets/24dce119-5e5a-4042-aaf5-91a559eb2828)
## The system operates through nested cognitive loops that implement a recursive market interpretation framework:

1. **Market Signal Perception**: Raw data ingestion and normalization
2. **Agent-Specific Processing**: Philosophy-aligned interpretation
3. **Multi-Agent Deliberation**: Signal exchange and position debate
4. **Recursive Arbitration**: Meta-agent integration and resolution
5. **Position Formulation**: Final decision synthesis with attribution
6. **Temporal Reflection**: Performance evaluation and belief updating

## **`Installation`**

```bash
# Clone the repository
git clone https://github.com/Multi-Agent Hedge Fund/Multi-Agent Hedge Fund.git
cd Multi-Agent Hedge Fund

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## **`Quick Start`**

```python
from multi_agent_debate import PortfolioManager
from multi_agent_debate.agents import GrahamAgent, WoodAgent, DalioAgent
from multi_agent_debate.market import MarketEnvironment

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

Multi-Agent Hedge Fund prioritizes transparent decision-making through recursive attribution tracing. Use the following flags to inspect agent cognition:

```bash
# Run with complete reasoning trace
python -m multi_agent_debate.run --show-trace

# Visualize agent consensus formation
python -m multi_agent_debate.run --consensus-graph

# Map conflicts in multi-agent deliberation
python -m multi_agent_debate.run --agent-conflict-map

# Generate attribution report for all trades
python -m multi_agent_debate.run --attribution-report
```

## **`Extending the Framework`**

The system is designed for extensibility at multiple levels:

### Creating Custom Agents

```python
from multi_agent_debate.agents import BaseAgent

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
from multi_agent_debate.cognition import ArbitrationMechanism

class CustomArbitration(ArbitrationMechanism):
    def __init__(self, weighting_strategy="confidence"):
        super().__init__(weighting_strategy=weighting_strategy)
        
    def resolve_conflicts(self, signals):
        # Implement custom conflict resolution logic
        resolution = self.recursive_integration(signals)
        return resolution
```

## 📄 License

This project is licensed under the PolyForm License - see the [LICENSE](LICENSE) file for details.

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

If you use Multi-Agent Hedge Fund in your research, please cite:

```bibtex
@software{multi_agent_debate2024,
  author = {{Multi-Agent Hedge Fund Contributors}},
  title = {Multi-Agent Hedge Fund: Multi-agent recursive market cognition framework},
  url = {https://github.com/Multi-Agent Hedge Fund/Multi-Agent Hedge Fund},
  year = {2024},
}
```

## 🌟 Acknowledgements

- The philosophical agents are inspired by the investment approaches of Benjamin Graham, Cathie Wood, Ray Dalio, Bill Ackman, Jim Simons, and Nassim Nicholas Taleb
- Recursive reasoning architecture influenced by work in multi-agent systems and interpretability research
- Market simulation components build upon open-source financial analysis libraries

---

<div align="center">
  <p>Built with ❤️ by the Multi-Agent Hedge Fund team</p>
  <p><i>Recursion. Interpretation. Emergence.</i></p>
</div>
