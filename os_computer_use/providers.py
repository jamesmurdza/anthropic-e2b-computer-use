import os
from dotenv import load_dotenv
from os_computer_use.llm_provider import LiteLLMBaseProvider
from os_computer_use.osatlas_provider import OSAtlasProvider
from os_computer_use.showui_provider import ShowUIProvider

# Load environment variables from .env file
load_dotenv()

# LLM providers use the OpenAI specification and require a base URL:


class LiteLLMProvider(LiteLLMBaseProvider):
    """Universal provider for all LLM models using LiteLLM"""

    VISION_MODELS = {
        "llama3.2": {
            "groq": "groq/llama-3.2-90b-vision-preview",
            "fireworks": "fireworks/llama-v3p2-90b-vision-instruct",
            "openrouter": "openrouter/meta-llama/llama-3.2-90b-vision-instruct",
        },
        "gpt-4-vision": "openai/gpt-4-vision-preview",
        "gemini-pro-vision": "gemini/gemini-pro-vision",
        "pixtral": "mistral/pixtral-large-latest",
        "gemini-2.0-flash-vision": "gemini/gemini-2.0-flash-vision",
    }

    TEXT_MODELS = {
        "llama3.3": {
            "groq": "groq/llama-3.3-70b-versatile",
            "fireworks": "fireworks/llama-v3p3-70b-instruct",
        },
        "small": "mistral/mistral-small-latest",
        "medium": "mistral/mistral-medium-latest",
        "large": "mistral/mistral-large-latest",
        "gpt-4": "openai/gpt-4-turbo-preview",
        "claude-3-opus": "anthropic/claude-3-opus-20240229",
        "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20241022",
        "claude-3-haiku": "anthropic/claude-3-haiku-20240229",
        "gemini-pro": "gemini/gemini-pro",
        "gemini-2.0-flash": "gemini/gemini-2.0-flash",
        "gemini-2.0-flash-exp": "gemini/gemini-2.0-flash-exp",
        "gemini-2.0-flash-lite-preview-02-05": "gemini/gemini-2.0-flash-lite-preview-02-05",
        "deepseek-coder": "deepseek/deepseek-coder-33b-instruct",
    }

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
        "gemini": {"provider": "gemini", "model_template": "gemini/{model}"},
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

        # Get provider mapping from PROVIDER_MODEL_MAPPINGS.
        parts = model_path.split("/")
        provider_name = parts[0]
        provider_config = self.PROVIDER_MODEL_MAPPINGS.get(
            provider_name,
            {"provider": provider_name, "model_template": f"{provider_name}/{{model}}"},
        )

        # Transform model path according to provider requirements.
        # This always formats the model as per the mapping.
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
