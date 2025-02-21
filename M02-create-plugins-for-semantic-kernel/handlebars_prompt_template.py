import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments, KernelFunctionFromPrompt
from semantic_kernel.prompt_template import HandlebarsPromptTemplate, PromptTemplateConfig

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
    <message role="system">Instructions: Identify the from and to destinations 
    and dates from the user's request</message>

    <message role="user">Can you give me a list of flights from Seattle to Tokyo? 
    I want to travel from March 11 to March 18.</message>

    <message role="assistant">
    Origin: Seattle
    Destination: Tokyo
    Depart: 03/11/2025 
    Return: 03/18/2025
    </message>

    <message role="user">{{input}}</message>
    """

input = "I want to travel from June 1 to July 22. I want to go to Greece. I live in Chicago."

arguments = KernelArguments(
    input = input
)

prompt_template_config = PromptTemplateConfig(
    template = prompt,
    template_format = "handlebars",
    name = "flight_search"
)
template_factory = HandlebarsPromptTemplate(
    prompt_template_config = prompt_template_config
)

function = KernelFunctionFromPrompt(
    prompt_template_config = prompt_template_config,
    prompt_template = template_factory,
    function_name = "flight_search"
)

async def invoke_kernel():
    # Obtener la respuesta del kernel
    response = await kernel.invoke(function, arguments)

    # Imprimir la respuesta
    print(response)

# Crear un bucle de eventos para ejecutar la función asincrónica
asyncio.run(invoke_kernel())  # Ejecutar la función asincrónica
