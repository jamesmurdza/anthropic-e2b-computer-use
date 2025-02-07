# Define the models to use in the agent

from os_computer_use import providers

grounding_model = providers.OSAtlasProvider()
# grounding_model = providers.ShowUIProvider()


# Custom Vision Models
# vision_model = providers.FireworksProvider("llama3.2")
# vision_model = providers.OpenAIProvider("gpt-4o")
# vision_model = providers.AnthropicProvider("claude-3.5-sonnet")
# vision_model = providers.GroqProvider("llama3.2")
# vision_model = providers.MistralProvider(
#     "pixtral"
# )  # pixtral-large-latest has vision capabilities


# Custom Action Models
# action_model = providers.FireworksProvider("llama3.3")
# action_model = providers.OpenAIProvider("gpt-4o")
# action_model = providers.AnthropicProvider("claude-3.5-sonnet")
# action_model = providers.GroqProvider("llama3.3")
# action_model = providers.MistralProvider(
#     "large"
# )  # mistral-large-latest for non-vision tasks


# Vision models using LiteLLM:
vision_model = providers.LiteLLMProvider("pixtral")  # Mistral
# vision_model = providers.LiteLLMProvider("llama3.2", provider="fireworks")  # Fireworks
# vision_model = providers.LiteLLMProvider("gpt-4-vision")  # OpenAI
# vision_model = providers.LiteLLMProvider("llama3.2", provider="groq")  # Groq
# vision_model = providers.LiteLLMProvider("claude-3-opus")  # Anthropic


# Action models using LiteLLM:
action_model = providers.LiteLLMProvider("large")  # Mistral
# action_model = providers.LiteLLMProvider("llama3.3", provider="fireworks")  # Fireworks
# action_model = providers.LiteLLMProvider("gpt-4")  # OpenAI
# action_model = providers.LiteLLMProvider("claude-3-sonnet")  # Anthropic
