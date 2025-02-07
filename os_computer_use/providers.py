import os
from dotenv import load_dotenv
from os_computer_use.llm_provider import (
    OpenAIBaseProvider,
    AnthropicBaseProvider,
    MistralBaseProvider,
    LiteLLMBaseProvider,
)
from os_computer_use.osatlas_provider import OSAtlasProvider
from os_computer_use.showui_provider import ShowUIProvider

# Load environment variables from .env file
load_dotenv()

# LLM providers use the OpenAI specification and require a base URL:


class LlamaProvider(OpenAIBaseProvider):
    base_url = "https://api.llama-api.com"
    api_key = os.getenv("LLAMA_API_KEY")
    aliases = {"llama3.2": "llama3.2-90b-vision", "llama3.3": "llama3.3-70b"}


class OpenRouterProvider(OpenAIBaseProvider):
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")
    aliases = {"llama3.2": "meta-llama/llama-3.2-90b-vision-instruct"}


class FireworksProvider(OpenAIBaseProvider):
    base_url = "https://api.fireworks.ai/inference/v1"
    api_key = os.getenv("FIREWORKS_API_KEY")
    aliases = {
        "llama3.2": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        "llama3.3": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    }


class DeepSeekProvider(OpenAIBaseProvider):
    base_url = "https://api.deepseek.com"
    api_key = os.getenv("DEEPSEEK_API_KEY")


class OpenAIProvider(OpenAIBaseProvider):
    base_url = "https://api.openai.com/v1"
    api_key = os.getenv("OPENAI_API_KEY")


class GeminiProvider(OpenAIBaseProvider):
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
    api_key = os.getenv("GEMINI_API_KEY")


class AnthropicProvider(AnthropicBaseProvider):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    aliases = {
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        "claude-3-opus": "claude-3-opus-20240229",
    }


class GroqProvider(OpenAIBaseProvider):
    base_url = "https://api.groq.com/openai/v1"
    api_key = os.getenv("GROQ_API_KEY")
    aliases = {
        "llama3.2": "llama-3.2-90b-vision-preview",
        "llama3.3": "llama-3.3-70b-versatile",
    }


class MistralProvider(MistralBaseProvider):
    base_url = "https://api.mistral.ai/v1"
    api_key = os.getenv("MISTRAL_API_KEY")
    aliases = {
        "small": "mistral-small-latest",
        "medium": "mistral-medium-latest",
        "large": "mistral-large-latest",
        "pixtral": "pixtral-large-latest",
    }


class LiteLLMProvider(LiteLLMBaseProvider):
    """Universal provider for all LLM models using LiteLLM"""

    # Model capabilities
    VISION_MODELS = {
        "llama3.2": {
            "groq": "groq/llama-3.2-90b-vision-preview",
            "fireworks": "fireworks/llama-v3p2-90b-vision-instruct",
            "openrouter": "openrouter/meta-llama/llama-3.2-90b-vision-instruct",
        },
        "gpt-4-vision": "openai/gpt-4-vision-preview",
        "gemini-pro-vision": "google/gemini-pro-vision",
        "pixtral": "mistral/pixtral-large-latest",
    }

    TEXT_MODELS = {
        "llama3.3": {
            "groq": "groq/llama-3.3-70b-versatile",
            "fireworks": "fireworks/llama-v3p3-70b-instruct",
        },
        # Mistral models
        "small": "mistral/mistral-small-latest",
        "medium": "mistral/mistral-medium-latest",
        "large": "mistral/mistral-large-latest",
        # OpenAI models
        "gpt-4": "openai/gpt-4-turbo-preview",
        # Anthropic models
        "claude-3-opus": "anthropic/claude-3-opus-20240229",
        "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
        "claude-3-haiku": "anthropic/claude-3-haiku-20240229",
        # Additional models
        "gemini-pro": "google/gemini-pro",
        "deepseek-coder": "deepseek/deepseek-coder-33b-instruct",
    }

    # Combined aliases for easy lookup
    aliases = {**VISION_MODELS, **TEXT_MODELS}

    # Model mappings for different providers
    PROVIDER_MODEL_MAPPINGS = {
        "fireworks": {
            "provider": "fireworks_ai",
            "model_template": "fireworks_ai/accounts/fireworks/models/{model}",
        },
        "openai": {"provider": "openai", "model_template": "openai/{model}"},
        "anthropic": {"provider": "anthropic", "model_template": "anthropic/{model}"},
        "groq": {"provider": "groq", "model_template": "groq/{model}"},
        "mistral": {"provider": "mistral", "model_template": "mistral/{model}"},
    }

    def __init__(self, model, provider=None):
        # If model has multiple providers, use specified provider or default
        model_info = self.aliases.get(model)
        if isinstance(model_info, dict):
            if not provider:
                # Default to first available provider
                provider = next(iter(model_info))
            model_path = model_info[provider]
        else:
            model_path = model_info

        # Get provider mapping
        provider_name = model_path.split("/")[0]
        provider_config = self.PROVIDER_MODEL_MAPPINGS.get(
            provider_name,
            {"provider": provider_name, "model_template": f"{provider_name}/{{model}}"},
        )

        # Transform model path according to provider requirements
        if "/" in model_path:
            _, model_name = model_path.split("/", 1)
            self.model = provider_config["model_template"].format(model=model_name)
            self.provider = provider_config["provider"]
        else:
            self.model = model_path
            self.provider = provider_name

        # Set API key based on provider
        self.api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
        super().__init__(self.model)

    @classmethod
    def get_vision_models(cls):
        """Get list of available vision models"""
        return list(cls.VISION_MODELS.keys())

    @classmethod
    def get_text_models(cls):
        """Get list of available text-only models"""
        return list(cls.TEXT_MODELS.keys())

    @classmethod
    def get_providers(cls, model):
        """Get available providers for a model"""
        model_info = cls.aliases.get(model)
        if isinstance(model_info, dict):
            return list(model_info.keys())
        return [model_info.split("/")[0]] if model_info else []
