# Define the models to use in the agent

from os_computer_use import providers

grounding_model = providers.OSAtlasProvider()
# grounding_model = providers.ShowUIProvider()


# Vision models using LiteLLM:
vision_model = providers.LiteLLMProvider("pixtral")  # Mistral
# vision_model = providers.LiteLLMProvider("llama3.2", provider="fireworks")  # Fireworks
# vision_model = providers.LiteLLMProvider("gpt-4-vision")  # OpenAI
# vision_model = providers.LiteLLMProvider("llama3.2", provider="groq")  # Groq
# vision_model = providers.LiteLLMProvider("claude-3-5-sonnet")  # Anthropic
# vision_model = providers.LiteLLMProvider("gemini-2.0-flash", provider="gemini")  # Gemini

# Action models using LiteLLM:
action_model = providers.LiteLLMProvider("large")  # Mistral
# action_model = providers.LiteLLMProvider("llama3.3", provider="fireworks")  # Fireworks
# action_model = providers.LiteLLMProvider("llama3.3", provider="groq")  # Groq
# action_model = providers.LiteLLMProvider("gpt-4")  # OpenAI
# action_model = providers.LiteLLMProvider("claude-3-5-sonnet")  # Anthropic
# action_model = providers.LiteLLMProvider("gemini-2.0-flash", provider="gemini")  # Gemini
