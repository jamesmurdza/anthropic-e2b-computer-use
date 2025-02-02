from os_computer_use.grounding import draw_big_dot, extract_bbox_midpoint
from os_computer_use.llm import (
    call_grounding_model,
    call_action_model,
    call_vision_model,
    Message,
    Text,
    Image as Base64Image,
)
from os_computer_use.utils import print_colored

import shlex
import os
import tempfile
from PIL import Image
import json

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

tools = {
    "stop": {
        "description": "Indicate that the task has been completed.",
        "params": {},
    }
}


class SandboxAgent:

    def __init__(self, sandbox):
        super().__init__()
        self.messages = []
        self.sandbox = sandbox
        self.latest_screenshot = None
        self.tmp_dir = tempfile.mkdtemp()
        self.image_counter = 0
        print("The agent will use the following actions:")
        for action, details in tools.items():
            param_str = ", ".join(details.get("params").keys())
            print(f"- {action}({param_str})")

    def call_function(self, name, arguments):

        func_impl = getattr(self, name.lower()) if name.lower() in tools else None
        if func_impl:
            try:
                result = func_impl(**arguments) if arguments else func_impl()
                return result
            except Exception as e:
                return f"Error executing function: {str(e)}"
        else:
            return "Function not implemented."

    def tool(description, params):
        def decorator(func):
            tools[func.__name__] = {"description": description, "params": params}
            return func

        return decorator

    def save_image(self, image, prefix="image"):
        self.image_counter += 1
        filename = f"{prefix}_{self.image_counter}.png"
        filepath = os.path.join(self.tmp_dir, filename)
        if isinstance(image, Image.Image):
            image.save(filepath)
        else:
            with open(filepath, "wb") as f:
                f.write(image)
        return filepath

    def take_screenshot(self):
        file = self.sandbox.take_screenshot()
        filename = self.save_image(file, "screenshot")
        print_colored(f"screenshot {filename}", color="gray")
        self.latest_screenshot = filename
        with open(filename, "rb") as image_file:
            return image_file.read()

    @tool(
        description="Run a shell command and return the result.",
        params={"command": "Shell command to run synchronously"},
    )
    def run_command(self, command):
        result = self.sandbox.commands.run(command, timeout=5)
        stdout, stderr = result.stdout, result.stderr
        if stdout and stderr:
            return stdout + "\n" + stderr
        elif stdout or stderr:
            return stdout + stderr
        else:
            return "The command finished running."

    @tool(
        description="Run a shell command in the background.",
        params={"command": "Shell command to run asynchronously"},
    )
    def run_background_command(self, command):
        self.sandbox.commands.run(command, background=True)
        return "The command has been started."

    @tool(
        description="Send a key or combination of keys to the system.",
        params={"name": "Key or combination (e.g. 'Return', 'Ctl-C')"},
    )
    def send_key(self, name):
        self.sandbox.commands.run(f"xdotool key -- {name}")
        return "The key has been pressed."

    @tool(
        description="Type a specified text into the system.",
        params={"text": "Text to type"},
    )
    def type_text(self, text):
        def chunks(text, n):
            for i in range(0, len(text), n):
                yield text[i : i + n]

        results = []
        for chunk in chunks(text, TYPING_GROUP_SIZE):
            cmd = f"xdotool type --delay {TYPING_DELAY_MS} -- {shlex.quote(chunk)}"
            results.append(self.sandbox.commands.run(cmd))
        return "The text has been typed."

    @tool(
        description="Click on a specified UI element.",
        params={"query": "Item or UI element on the screen to click"},
    )
    def click(self, query):
        self.take_screenshot()
        bbox, image_url = call_grounding_model(
            query + "\nReturn the response in the form of a bbox",
            self.latest_screenshot,
        )
        position = extract_bbox_midpoint(bbox)
        print_colored(f"bbox {image_url}", color="gray")

        dot_image = draw_big_dot(Image.open(self.latest_screenshot), position)
        filepath = self.save_image(dot_image, "location")
        print_colored(f"click {filepath})", color="gray")

        x, y = position
        self.sandbox.commands.run(f"xdotool mousemove --sync {x} {y}")
        self.sandbox.commands.run("xdotool click 1")
        return "The mouse has been clicked."

    def append_screenshot(self):
        convert_to_content = lambda message: (
            Base64Image(message) if isinstance(message, bytes) else Text(message)
        )
        return call_vision_model(
            [
                *self.messages,
                Message(
                    map(
                        convert_to_content,
                        [
                            self.take_screenshot(),
                            "This image shows the current display of the computer. Please respond in the following format:\n"
                            "The objective is: [put the objective here]\n"
                            "On the screen, I see: [an extensive list of everything that might be relevant to the objective including windows, icons, menus, apps, and UI elements]\n"
                            "This means the objective is: [complete|not complete]\n\n"
                            "(Only continue if the objective is not complete.)\n"
                            "The next step is to [click|type|run the shell command] [put the next single step here] in order to [put what you expect to happen here].",
                        ],
                    ),
                    role="user",
                    log=False,
                ),
            ]
        )

    def run(self, instruction):

        self.messages.append(Message(f"OBJECTIVE: {instruction}", log=False))

        should_continue = True
        while should_continue:
            # Stop the sandbox from timing out
            self.sandbox.set_timeout(60)

            content, tool_calls = call_action_model(
                [
                    Message(
                        "You are an AI assistant with computer use abilities.",
                        role="system",
                        log=False,
                    ),
                    *self.messages,
                    Message(f"THOUGHT: {self.append_screenshot()}", color="green"),
                    Message(
                        "I will now use tool calls to take these actions, or use the stop command if the objective is complete.",
                        log=False,
                    ),
                ],
                tools,
            )

            if content:
                self.messages.append(Message(f"THOUGHT: {content}", color="blue"))

            should_continue = False
            for tool_call in tool_calls:
                name, parameters = tool_call.get("name"), tool_call.get("parameters")
                should_continue = name != "stop"
                if not should_continue:
                    break
                # Print the tool-call in an easily readable format
                print_colored(f"ACTION: {name} {str(parameters)}", color="red")
                # Write the tool-call to the message history using the same format used by the model
                self.messages.append(Message(json.dumps(tool_call), log=False))
                result = self.call_function(name, parameters)

                self.messages.append(Message(f"OBSERVATION: {result}", color="yellow"))
