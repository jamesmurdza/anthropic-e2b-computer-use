from os_computer_use.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GroqProvider,
    FireworksProvider,
    MistralProvider,
    LiteLLMProvider,
)
from os_computer_use.llm_provider import Message


# Define tools available for use
tools = {
    "click_item": {
        "description": "Click on an item on the screen",
        "params": {"description": "Description of the item to click on"},
    }
}


# Function to simulate taking a screenshot
def take_screenshot():
    with open("./tests/test_screenshot.png", "rb") as f:
        return f.read()


# Prompt to test tool calls with vision
toolcall_messages = [
    Message(
        [
            "You can use tools to operate the computer. Take the next step to Google.com",
            take_screenshot(),
        ],
        role="user",
    )
]

# Prompt to test vision
messages = [
    Message(
        [
            "Describe what you see in the image below.",
            take_screenshot(),
        ],
        role="user",
    )
]

# Anthropic
opus = AnthropicProvider("claude-3-opus")
print(opus.call(toolcall_messages, tools)[1])
print(opus.call(messages))

# OpenAI
gpt4o = OpenAIProvider("gpt-4o")
print(gpt4o.call(toolcall_messages, tools)[1])
print(gpt4o.call(messages))

# Groq
groq = GroqProvider("llama3.2")
print(groq.call(toolcall_messages, tools)[1])
print(groq.call(messages))

# Fireworks
fireworks = FireworksProvider("llama3.2")
print(fireworks.call(toolcall_messages, tools)[1])
print(fireworks.call(messages))


# Pixtral
mistral = MistralProvider("pixtral")
print("\nTesting Mistral :")
print(mistral.call(toolcall_messages, tools)[1])
print(mistral.call(messages))


# Mistral Large (non-vision) using text-only messages
mistral_large = MistralProvider(
    "large"
)  # Using mistral-large-latest for non-vision tasks
text_messages = [Message("What is the capital of France?", role="user")]
print("\nTesting Mistral Large with text-only:")
print(mistral_large.call(text_messages))

# Test tool calls for Mistral Large using text-only messages (no image data)
text_tool_messages = [Message("Click on the submit button", role="user")]
print("\nTesting Mistral Large Tool Calls with text:")
print(mistral_large.call(text_tool_messages, tools)[1])


# Test LiteLLM with Pixtral
litellm = LiteLLMProvider("pixtral")
print("\nTesting LiteLLM with Pixtral:")
print(litellm.call(toolcall_messages, tools)[1])
print(litellm.call(messages))

# Test LiteLLM with Mistral Large (non-vision)
litellm_large = LiteLLMProvider("large")  # Using mistral-large-latest
text_messages = [Message("What is the capital of France?", role="user")]
print("\nTesting LiteLLM Mistral Large with text-only:")
print(litellm_large.call(text_messages))

# Test tool calls for Mistral Large using text-only messages
text_tool_messages = [Message("Click on the submit button", role="user")]
print("\nTesting LiteLLM Mistral Large Tool Calls with text:")
print(litellm_large.call(text_tool_messages, tools)[1])
