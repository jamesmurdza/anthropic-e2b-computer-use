from os_computer_use.providers import (
    AnthropicProvider,
    OpenAIProvider,
    GroqProvider,
    FireworksProvider,
    MistralProvider,
    LiteLLMProvider,
)
from os_computer_use.llm_provider import Message
import base64


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

# Test LiteLLM with Claude-3-Opus
litellm_claude = LiteLLMProvider("claude-3-opus")
print("\nTesting LiteLLM with Claude-3-Opus:")
print(litellm_claude.call(toolcall_messages, tools)[1])
print(litellm_claude.call(messages))


# Test LiteLLM with Claude-3-Sonnet
litellm_claude_sonnet = LiteLLMProvider("claude-3-sonnet")
print("\nTesting LiteLLM with Claude-3-Sonnet:")
print(litellm_claude_sonnet.call(toolcall_messages, tools)[1])
print(litellm_claude_sonnet.call(messages))


# Test LiteLLM with Gemini 2.0
gemini_llm = LiteLLMProvider("gemini-2.0-flash")
print("\nTesting LiteLLM with Gemini-2.0-Flash:")
print(gemini_llm.call(toolcall_messages, tools)[1])
print(gemini_llm.call(messages))
