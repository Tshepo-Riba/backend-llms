from tools.coding_execution_tools import ToolsFactory
from tools.langchain._tools import WebSearchTool
from backend import Client

############# Running Tools ############################
# Create an instance of the DockerPythonREPLExecutor
executor = ToolsFactory.python_repl_executor(
    image="python:3.10-slim", execution_timeout=30
)

# Create an instance of the PythonREPLTool
# python_repl_tool = PythonREPLTool(execution_timeout=30)

# Prepare the arguments for the Python code
arguments = {
    "code": "print('Hello, world!');",
}

# Execute the Python code using the DockerPythonREPLExecutor
result, stdout, stderr = executor.execute_code(arguments, use_docker=False)

# Print the result, stdout, and stderr
# print("Result Variables:", result)
print("Standard Output:", stdout)
# print("Standard Error:", stderr)
web_search_tool = WebSearchTool()
arguments = {"engine": "duckduckgo", "query": "python programming"}
result = web_search_tool.run(arguments)

print(result)

######################## Running LLM API's ###############################


client = Client(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


response = client.create(
    model="togetherai/Llama-3-70b-chat-hf",
    messages=[
        {"role": "user", "content": "What's the weather like in Boston today?"},
        {
            "role": "system",
            "content": "You are an AI assistant for business research. Your responses should be informative and concise.",
        },
    ],
    stream=True,
    max_tokens=500,
    tools=tools,
)

print(response)
print(client.messages(response))
print(client.cost(response))
print(client.usage(response))
