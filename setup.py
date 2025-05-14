from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multi_agent_debate",
    version="0.1.0",
    author="multi_agent_debate Contributors",
    author_email="contributors@multi_agent_debate.org",
    description="Multi-agent recursive market cognition framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/multi_agent_debate/multi_agent_debate",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "yfinance>=0.2.18",
        "langgraph>=0.0.11",
        "anthropic>=0.5.0",
        "openai>=1.1.0",
        "groq>=0.3.0",
        "langchain>=0.0.267",
        "langchain-experimental>=0.0.9", 
        "langchain-community>=0.0.9",
        "pydantic>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.1",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.2",
            "myst-parser>=2.0.0",
        ],
        "optional": [
            "polygon-api-client>=1.10.0",
            "alpha_vantage>=2.3.1",
            "ollama>=0.1.0",
            "deepseek>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "multi_agent_debate=src.main:main",
        ],
    },
)
