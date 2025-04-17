"""
ModelRouter - Multi-Model LLM Orchestration Framework

This module implements the model routing architecture that allows agents to 
interact with various LLM providers including:
- OpenAI (GPT models)
- Anthropic (Claude models)
- Groq (Llama and Mixtral)
- Ollama (local models)
- DeepSeek (DeepSeek models)

Key capabilities:
- Provider-agnostic interface for agent interactions
- Dynamic provider selection based on capabilities
- Fallback chains for reliability
- Prompt template management with provider-specific optimizations
- Token usage tracking and optimization
- Output parsing and normalization

Internal Note: The model router implements a symbolic abstraction layer over
different LLM providers while maintaining a unified attribution interface.
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
import traceback
from enum import Enum
from abc import ABC, abstractmethod

# Optional imports for different providers
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import groq
except ImportError:
    groq = None

try:
    import ollama
except ImportError:
    ollama = None


class ModelCapability(Enum):
    """Capabilities that models may support."""
    REASONING = "reasoning"             # Complex multi-step reasoning
    CODE_GENERATION = "code_generation" # Code writing and analysis
    FINANCE = "finance"                 # Financial analysis and modeling
    RAG = "rag"                         # Retrieval augmented generation
    TOOL_USE = "tool_use"               # Using external tools
    FUNCTION_CALLING = "function_calling" # Structured function calling
    JSON_MODE = "json_mode"             # Reliable JSON output


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        if not self.api_key:
            logging.warning("OpenAI API key not provided. OpenAI provider will not work.")
        
        if openai is None:
            logging.warning("OpenAI Python package not installed. OpenAI provider will not work.")
        
        # Initialize client if possible
        self.client = None
        if openai is not None and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        
        # Define models and capabilities
        self.models = {
            "gpt-4-0125-preview": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
            ],
            "gpt-4-turbo-preview": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
            ],
            "gpt-4": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.FUNCTION_CALLING,
            ],
            "gpt-3.5-turbo": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
            ],
            "gpt-3.5-turbo-instruct": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using OpenAI.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - model: Model name (default: gpt-4-turbo-preview)
                - temperature: Temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
                - json_mode: Whether to enforce JSON output (default: False)
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Provide a valid API key.")
        
        # Extract parameters with defaults
        model = kwargs.get("model", "gpt-4-turbo-preview")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        json_mode = kwargs.get("json_mode", False)
        
        try:
            # Create messages
            messages = [{"role": "user", "content": prompt}]
            
            # Create parameters
            params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add response format if JSON mode
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            
            # Make request
            response = self.client.chat.completions.create(**params)
            
            # Extract text
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating text with OpenAI: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        return self.models.get(model_name, [])


class AnthropicProvider(ModelProvider):
    """Anthropic model provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            logging.warning("Anthropic API key not provided. Anthropic provider will not work.")
        
        if anthropic is None:
            logging.warning("Anthropic Python package not installed. Anthropic provider will not work.")
        
        # Initialize client if possible
        self.client = None
        if anthropic is not None and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Define models and capabilities
        self.models = {
            "claude-3-opus-20240229": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.JSON_MODE,
            ],
            "claude-3-sonnet-20240229": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
                ModelCapability.TOOL_USE,
                ModelCapability.JSON_MODE,
            ],
            "claude-3-haiku-20240307": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
                ModelCapability.JSON_MODE,
            ],
            "claude-2.1": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
            ],
            "claude-2.0": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
            ],
            "claude-instant-1.2": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using Anthropic.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - model: Model name (default: claude-3-sonnet-20240229)
                - temperature: Temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
                - system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("Anthropic client not initialized. Provide a valid API key.")
        
        # Extract parameters with defaults
        model = kwargs.get("model", "claude-3-sonnet-20240229")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        system_prompt = kwargs.get("system_prompt", "")
        
        try:
            # Create parameters
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Add system prompt if provided
            if system_prompt:
                params["system"] = system_prompt
            
            # Make request
            response = self.client.messages.create(**params)
            
            # Extract text
            return response.content[0].text
            
        except Exception as e:
            logging.error(f"Error generating text with Anthropic: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        return self.models.get(model_name, [])


class GroqProvider(ModelProvider):
    """Groq model provider."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq provider.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            logging.warning("Groq API key not provided. Groq provider will not work.")
        
        if groq is None:
            logging.warning("Groq Python package not installed. Groq provider will not work.")
        
        # Initialize client if possible
        self.client = None
        if groq is not None and self.api_key:
            self.client = groq.Groq(api_key=self.api_key)
        
        # Define models and capabilities
        self.models = {
            "llama2-70b-4096": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
            ],
            "mixtral-8x7b-32768": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
            ],
            "gemma-7b-it": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using Groq.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - model: Model name (default: mixtral-8x7b-32768)
                - temperature: Temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("Groq client not initialized. Provide a valid API key.")
        
        # Extract parameters with defaults
        model = kwargs.get("model", "mixtral-8x7b-32768")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        try:
            # Create parameters
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # Make request
            response = self.client.chat.completions.create(**params)
            
            # Extract text
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating text with Groq: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        return self.models.get(model_name, [])


class OllamaProvider(ModelProvider):
    """Ollama model provider for local models."""
    
    def __init__(self, host: str = "http://localhost:11434"):
        """
        Initialize Ollama provider.
        
        Args:
            host: Ollama host address (default: http://localhost:11434)
        """
        self.host = host
        
        if ollama is None:
            logging.warning("Ollama Python package not installed. Ollama provider will not work.")
        
        # Check if Ollama is available
        self.available = False
        if ollama is not None:
            try:
                # Try to list models
                self._list_models()
                self.available = True
            except Exception as e:
                logging.warning(f"Ollama not available: {e}")
        
        # Define models and capabilities
        self.models = {
            "llama3:latest": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
            "mistral:latest": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
            "codellama:latest": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
            "deepseek-coder:latest": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
            "wizardcoder:latest": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
            "gemma2:latest": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
        }
    
    def _list_models(self) -> List[str]:
        """List available Ollama models."""
        if ollama is None:
            return []
        
        try:
            response = ollama.list(api_base=self.host)
            return [model['name'] for model in response['models']]
        except Exception as e:
            logging.error(f"Error listing Ollama models: {e}")
            return []
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using Ollama.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - model: Model name (default: mistral:latest)
                - temperature: Temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
            
        Returns:
            Generated text
        """
        if ollama is None:
            raise ValueError("Ollama package not installed.")
        
        if not self.available:
            raise ValueError("Ollama not available.")
        
        # Extract parameters with defaults
        model = kwargs.get("model", "mistral:latest")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        try:
            # Make request
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                num_predict=max_tokens,
                api_base=self.host,
            )
            
            # Extract text
            return response['message']['content']
            
        except Exception as e:
            logging.error(f"Error generating text with Ollama: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if not self.available:
            return []
        
        return self._list_models()
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        # For unknown models, assume basic capabilities
        if model_name not in self.models:
            return [ModelCapability.RAG]
        
        return self.models.get(model_name, [])


class DeepSeekProvider(ModelProvider):
    """DeepSeek model provider using OpenAI-compatible API."""
    
    def __init__(self, api_key: Optional[str] = None, api_base: str = "https://api.deepseek.com/v1"):
        """
        Initialize DeepSeek provider.
        
        Args:
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            api_base: DeepSeek API base URL
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.api_base = api_base
        
        if not self.api_key:
            logging.warning("DeepSeek API key not provided. DeepSeek provider will not work.")
        
        if openai is None:
            logging.warning("OpenAI Python package not installed. DeepSeek provider will not work.")
        
        # Initialize client if possible
        self.client = None
        if openai is not None and self.api_key:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )
        
        # Define models and capabilities
        self.models = {
            "deepseek-chat": [
                ModelCapability.REASONING,
                ModelCapability.CODE_GENERATION,
                ModelCapability.FINANCE,
                ModelCapability.RAG,
            ],
            "deepseek-coder": [
                ModelCapability.CODE_GENERATION,
                ModelCapability.RAG,
            ],
        }
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt using DeepSeek.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
                - model: Model name (default: deepseek-chat)
                - temperature: Temperature (default: 0.7)
                - max_tokens: Maximum tokens (default: 2000)
            
        Returns:
            Generated text
        """
        if self.client is None:
            raise ValueError("DeepSeek client not initialized. Provide a valid API key.")
        
        # Extract parameters with defaults
        model = kwargs.get("model", "deepseek-chat")
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 2000)
        
        try:
            # Create messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make request
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract text
            return response.choices[0].message.content
            
        except Exception as e:
            logging.error(f"Error generating text with DeepSeek: {e}")
            logging.error(traceback.format_exc())
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())
    
    def get_model_capabilities(self, model_name: str) -> List[ModelCapability]:
        """Get capabilities of a specific model."""
        return self.models.get(model_name, [])


class ModelRouter:
    """
    Model router for multi-provider LLM orchestration.
    
    The ModelRouter provides:
    - Unified interface for multiple LLM providers
    - Dynamic provider selection based on capabilities
    - Fallback chains for reliability
    - Prompt template management
    - Attribution tracing for interpretability
    """
    
    def __init__(
        self,
        provider: str = "anthropic",
        model: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
    ):
        """
        Initialize model router.
        
        Args:
            provider: Default provider
            model: Default model (provider-specific)
            fallback_providers: List of fallback providers
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            groq_api_key: Groq API key
        """
        self.default_provider = provider
        self.default_model = model
        self.fallback_providers = fallback_providers or []
        
        # Track usage
        self.usage_stats = {
            "total_calls": 0,
            "total_tokens": 0,
            "provider_calls": {},
            "model_calls": {},
            "errors": {},
        }
        
        # Initialize providers
        self.providers = {}
        
        # Initialize OpenAI
        try:
            self.providers["openai"] = OpenAIProvider(api_key=openai_api_key)
            
            # Set default model if not specified
            if provider == "openai" and model is None:
                self.default_model = "gpt-4-turbo-preview"
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI provider: {e}")
        
        # Initialize Anthropic
        try:
            self.providers["anthropic"] = AnthropicProvider(api_key=anthropic_api_key)
            
            # Set default model if not specified
            if provider == "anthropic" and model is None:
                self.default_model = "claude-3-sonnet-20240229"
        except Exception as e:
            logging.warning(f"Failed to initialize Anthropic provider: {e}")
        
        # Initialize Groq
        try:
            self.providers["groq"] = GroqProvider(api_key=groq_api_key)
            
            # Set default model if not specified
            if provider == "groq" and model is None:
                self.default_model = "mixtral-8x7b-32768"
        except Exception as e:
            logging.warning(f"Failed to initialize Groq provider: {e}")
        
        # Initialize Ollama
        try:
            self.providers["ollama"] = OllamaProvider()
            
            # Set default model if not specified
            if provider == "ollama" and model is None:
                self.default_model = "mistral:latest"
        except Exception as e:
            logging.warning(f"Failed to initialize Ollama provider: {e}")
        
        # Initialize DeepSeek
        try:
            self.providers["deepseek"] = DeepSeekProvider()
            
            # Set default model if not specified
            if provider == "deepseek" and model is None:
                self.default_model = "deepseek-chat"
        except Exception as e:
            logging.warning(f"Failed to initialize DeepSeek provider: {e}")
        
        # Verify default provider is available
        if self.default_provider not in self.providers:
            available_providers = list(self.providers.keys())
            if available_providers:
                logging.warning(f"Default provider '{self.default_provider}' not available. "
                              f"Using '{available_providers[0]}' instead.")
                self.default_provider = available_providers[0]
            else:
                raise ValueError("No LLM providers available. Check API keys and dependencies.")
    
    def generate(self, prompt: str, provider: Optional[str] = None, 
               model: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            provider: Provider to use (default is instance default)
            model: Model to use (default is instance default)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        # Use default provider if not specified
        provider = provider or self.default_provider
        
        # Use default model if not specified
        model = model or self.default_model
        
        # Update usage stats
        self.usage_stats["total_calls"] += 1
        
        # Update provider stats
        if provider not in self.usage_stats["provider_calls"]:
            self.usage_stats["provider_calls"][provider] = 0
        self.usage_stats["provider_calls"][provider] += 1
        
        # Update model stats
        model_key = f"{provider}:{model}"
        if model_key not in self.usage_stats["model_calls"]:
            self.usage_stats["model_calls"][model_key] = 0
        self.usage_stats["model_calls"][model_key] += 1
        
        # Check if provider is available
        if provider not in self.providers:
            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                if fallback_provider in self.providers:
                    logging.warning(f"Provider '{provider}' not available. "
                                  f"Using fallback provider '{fallback_provider}'.")
                    return self.generate(prompt, provider=fallback_provider, model=model, **kwargs)
            
            # No fallback providers available
            raise ValueError(f"Provider '{provider}' not available and no fallback providers available.")
        
        try:
            # Get provider
            provider_instance = self.providers[provider]
            
            # Add model to kwargs
            if model:
                kwargs["model"] = model
            
            # Generate text
            start_time = time.time()
            response = provider_instance.generate(prompt, **kwargs)
            end_time = time.time()
            
            # Log generation time
            logging.debug(f"Generated text with {provider}:{model} in {end_time - start_time:.2f} seconds.")
            
            return response
            
        except Exception as e:
            # Update error stats
            error_key = str(type(e).__name__)
            if error_key not in self.usage_stats["errors"]:
                self.usage_stats["errors"][error_key] = 0
            self.usage_stats["errors"][error_key] += 1
            
            # Try fallback providers
            for fallback_provider in self.fallback_providers:
                if fallback_provider in self.providers:
                    logging.warning(f"Error with provider '{provider}': {e}. "
                                  f"Using fallback provider '{fallback_provider}'.")
                    return self.generate(prompt, provider=fallback_provider, model=model, **kwargs)
            
            # No fallback providers available
            logging.error(f"Error generating text with {provider}:{model}: {e}")
            logging.error(traceback.format_exc())
            raise
    
    async def generate_async(self, prompt: str, provider: Optional[str] = None, 
                         model: Optional[str] = None, **kwargs) -> str:
        """
        Generate text from prompt asynchronously.
        
        Args:
            prompt: Input prompt
            provider: Provider to use (default is instance default)
            model: Model to use (default is instance default)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text
        """
        # Async implementation using run_in_executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.generate(prompt, provider, model, **kwargs)
        )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())
    
    def get_available_models(self, provider: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Get available models for all providers or a specific provider.
        
        Args:
            provider: Optional provider to get models for
            
        Returns:
            Dictionary mapping providers to lists of models
        """
        if provider:
            if provider not in self.providers:
                return {}
            
            return {provider: self.providers[provider].get_available_models()}
        
        # Get models for all providers
        models = {}
        for provider_name, provider_instance in self.providers.items():
            models[provider_name] = provider_instance.get_available_models()
        
        return models
    
    def get_model_capabilities(self, provider: str, model: str) -> List[ModelCapability]:
        """
        Get capabilities of a specific model.
        
        Args:
            provider: Provider name
            model: Model name
            
        Returns:
            List of model capabilities
        """
        if provider not in self.providers:
            return []
        
        return self.providers[provider].get_model_capabilities(model)
    
    def find_models_with_capabilities(self, capabilities: List[ModelCapability]) -> List[Tuple[str, str]]:
        """
        Find models that have all specified capabilities.
        
        Args:
            capabilities: List of required capabilities
            
        Returns:
            List of (provider, model) tuples that have all capabilities
        """
        matching_models = []
        
        for provider_name, provider_instance in self.providers.items():
            for model in provider_instance.get_available_models():
                model_capabilities = provider_instance.get_model_capabilities(model)
                
                # Check if model has all required capabilities
                if all(capability in model_capabilities for capability in capabilities):
                    matching_models.append((provider_name, model))
        
        return matching_models
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_stats.copy()
