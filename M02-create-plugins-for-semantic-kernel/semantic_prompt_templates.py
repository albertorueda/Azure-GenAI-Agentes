import asyncio

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt

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
kernel.add_service(AzureChatCompletion(
    deployment_name = model_id, 
    endpoint = endpoint, 
    api_key = api_key
    )
)

prompt = """
    You are a helpful travel guide. 
    I'm visiting {{$city}}. {{$background}}. What are some activities I should do today?
    """

city = "Paris"
background = "I love art and history"

activities_function = KernelFunctionFromPrompt(
    prompt = prompt, 
    function_name = "activities"
)

arguments = KernelArguments(
    city = city,
    background = background
)

async def invoke_kernel():
    # Obtener la respuesta del kernel
    response = await kernel.invoke(activities_function, arguments)

    # Imprimir la respuesta
    print(response)

# Crear un bucle de eventos para ejecutar la función asincrónica
asyncio.run(invoke_kernel())  # Ejecutar la función asincrónica
