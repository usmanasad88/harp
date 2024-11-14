from litellm import completion
import litellm 
import json
## [OPTIONAL] REGISTER MODEL - not all ollama models support function calling, litellm defaults to json mode tool calls if native tool calling not supported.

litellm.register_model(model_cost={
                 "ollama_chat/llama3.1": { 
                   "supports_function_calling": True
                 },
             })

tools = [
  {
    "type": "function",
    "function": {
      "name": "robot_action",
      "description": "What action should a robot take next, based on the user input. If the input is disambigious or unrelated, then the action should be NoChange",
      "parameters": {
        "type": "object",
        "properties": {
          "direction": {
            "type": "string",
            "description": "The direction of motion or rotation.",
            "enum": ["Forward", "Backward","RotateLeft","RotateRight","Stop","NoChange"]
          },
        },
        "required": ["direction"],
      },
    }
  }
]

messages = [{"role": "user", "content": "Move towards me. Straight ahead"}]


response = completion(
  model="ollama_chat/llama3.1",
  messages=messages,
  tools=tools
)

#print(response)

def robot_action(direction):
    
    # Example implementation (you would replace this with actual API calls)
    print(direction)
    return direction

if isinstance(response, litellm.ModelResponse):
    # Extract the function call details from the response
    tool_call = response.choices[0].message.tool_calls[0]
    function_arguments = tool_call.function.arguments

    # Parse the function arguments
    function_args = json.loads(function_arguments)
    direction = function_args.get('direction', 'Stop')

# Example usage
GoDirection = robot_action(direction)
