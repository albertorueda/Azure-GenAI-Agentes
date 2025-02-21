import asyncio

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents import ChatHistory

import dotenv 
import os

# Load environment variables from .env file
dotenv.load_dotenv()

# Populate values from your OpenAI deployment
model_id = "gpt-35-turbo-16k"
endpoint = os.getenv("OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")

# Create a kernel with Azure OpenAI chat completion
kernel = Kernel()
chat_completion = AzureChatCompletion(
    deployment_name = model_id, 
    endpoint = endpoint, 
    api_key = api_key
    )
kernel.add_service(chat_completion)

# Enable planning
execution_settings = AzureChatPromptExecutionSettings()

chat_history = ChatHistory()
chat_history.add_system_message("Welcome to the chatbot!")

async def invoke_kernel(user_input):
    chat_history.add_user_message(user_input)
    
    # Get the response from the AI
    result = await chat_completion.get_chat_message_content(
        chat_history=chat_history,
        kernel=kernel,
        settings=execution_settings
    )

    # Print the results
    print("Bot: " + str(result))

    # Add the message from the agent to the chat history
    chat_history.add_message(result)

async def main():
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        await invoke_kernel(user_input)

# Ejecutar el bucle principal en un entorno asincrónico
asyncio.run(main())