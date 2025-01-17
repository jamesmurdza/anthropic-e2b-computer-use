import os
from dotenv import load_dotenv
from os_computer_use.llm_provider import LLMProvider
from os_computer_use.osatlas_provider import OSAtlasProvider, ShowUIProvider

# Load environment variables from .env file
load_dotenv()

# Model names can vary from provider to provider, and are standardized here:
model_names = {
    "llama": {"llama3.2": "llama3.2-90b-vision", "llama3.3": "llama3.3-70b"},
    "openrouter": {
        "llama3.2": "meta-llama/llama-3.2-90b-vision-instruct",
    },
    "fireworks": {
        "llama3.2": "accounts/fireworks/models/llama-v3p2-90b-vision-instruct",
        "llama3.3": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    },
    "deepseek": {"deepseek-chat": "deepseek-chat"},
    # grounding models support: showui, osatlas
    "grounder": "showui",
}


# LLM providers use the OpenAI specification and require a base URL:


class LlamaProvider(LLMProvider):
    base_url = "https://api.llama-api.com"
    api_key = os.getenv("LLAMA_API_KEY")


class OpenRouterProvider(LLMProvider):
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY")


class FireworksProvider(LLMProvider):
    base_url = "https://api.fireworks.ai/inference/v1"
    api_key = os.getenv("FIREWORKS_API_KEY")


class DeepSeekProvider(LLMProvider):
    base_url = "https://api.deepseek.com"
    api_key = os.getenv("DEEPSEEK_API_KEY")


# Define the models to use in the agent

if model_names["grounder"] == "showui":
    grounding_model = ShowUIProvider()
elif model_names["grounder"] == "osatlas":
    grounding_model = OSAtlasProvider()
else:
    raise NotImplementedError(f"Grounder: {model_names["grounder"]} not implemented yet.")

vision_model = FireworksProvider(model_names["fireworks"]["llama3.2"])
action_model = FireworksProvider(model_names["fireworks"]["llama3.3"])
