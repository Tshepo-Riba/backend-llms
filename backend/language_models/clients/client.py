import copy
import json
import time
import requests
from typing import Any, Dict, List, Optional, Union
from ...base.base import LLM
from .utilities.utils import (
    update_kwargs_with_config,
    remove_unwanted_params,
)
from ..configurations.models import configurations, embed_configurations

try:
    import google.generativeai as genai
    import google.generativeai as palm
except ImportError:
    raise Exception(
        "Failed to import google.generativeai. Please ensure you have installed it by running pip install -q google-generativeai"
    )


from ..clients import (
    AI21,
    ALEPHALPHA,
    ANTHROPIC,
    AZURE,
    BEDROCK,
    COHERE,
    GROQ,
    GOOGLE,
    HUGGINGFACE,
    OLLAMA,
    OPENAI,
    PERPLEXITY,
    SAGEMAKER,
    TOGETHERAI,
)


client_mapping = {
    "AI21": AI21,
    "ALEPHALPHA": ALEPHALPHA,
    "ANTHROPIC": ANTHROPIC,
    "AZURE": AZURE,
    "BEDROCK": BEDROCK,
    "COHERE": COHERE,
    "GROQ": GROQ,
    "HUGGINGFACE": HUGGINGFACE,
    "OLLAMA": OLLAMA,
    "OPENAI": OPENAI,
    "PERPLEXITY": PERPLEXITY,
    "SAGEMAKER": SAGEMAKER,
    "TOGETHERAI": TOGETHERAI,
    "GOOGLE": GOOGLE,
}


class Client(LLM):

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.api_key = api_key
        self.base_url = base_url.lower() if base_url else None
        self.organization = organization.lower() if organization else None
        self.config = config or {}
        self.__dict__.update(kwargs)

    def create(
        self,
        model: Optional[str],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Creates a chat completion based on the provided model and keyword arguments.

        Args:
            model (str): The model to use for chat completion.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Dict[str, Any]: The response model.

        Raises:
            ValueError: If model is missing or both prompt and messages are provided.
        """
        kwargs = {**self.config, **kwargs}
        model = model or kwargs.get("model")
        prompt = kwargs.get("prompt")
        messages = kwargs.get("messages")
        provider, model = model.split("/")

        if not model:
            raise ValueError("Please ensure that a model is provided.")

        if (messages is not None and prompt is not None) or (
            messages is None and prompt is None
        ):
            raise ValueError(
                "Please provide either a 'message' or a 'prompt', but not both."
            )
        if provider in ("bedrock", "sagemaker", "gemini", "palm", "vertex"):
            updated_kwargs = copy.deepcopy(kwargs)
        else:
            updated_kwargs = update_kwargs_with_config(provider, **kwargs)

        updated_kwargs = remove_unwanted_params(**updated_kwargs)
        # Dictionary mapping providers to client instantiation methods
        provider_clients = {
            "ai21": (AI21, ["api_key"]),
            "alephalpha": (ALEPHALPHA, ["token"]),
            "anthropic": (
                ANTHROPIC,
                [
                    "api_key",
                    "aws_access_key",
                    "aws_secret_key",
                    "aws_session_token",
                    "aws_region",
                    "project_id",
                    "region",
                ],
            ),
            "azure": (
                AZURE,
                [
                    "api_key",
                    "azure_ad_token_provider",
                    "deployment_name",
                    "base_url",
                    "api_version",
                    "azure_endpoint",
                ],
            ),
            "cohere": (COHERE, ["api_key"]),
            "groq": (GROQ, ["api_key"]),
            "ollama": (OLLAMA, []),
            "openai": (OPENAI, ["api_key", "base_url"]),
            "perplexity": (PERPLEXITY, ["api_key"]),
            "togetherai": (TOGETHERAI, ["api_key"]),
            "sagemaker": (
                SAGEMAKER,
                ["aws_access_key_id", "aws_secret_access_key", "aws_region_name"],
            ),
            "bedrock": (
                BEDROCK,
                [
                    "aws_role_name",
                    "aws_session_name",
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_region_name",
                    "aws_bedrock_runtime_endpoint",
                    "timeout",
                ],
            ),
            "huggingface": (HUGGINGFACE, ["api_key"]),
            "gemini": (GOOGLE, ["api_key"]),
        }

        # Check if provider is in the dictionary and get the client instantiation method and required kwargs
        if provider in provider_clients:
            client_class, required_kwargs = provider_clients[provider]
            kwargs_to_pass = {
                key: getattr(self, key) for key in required_kwargs if hasattr(self, key)
            }
            client = client_class(**kwargs_to_pass)

            # Build the response
            response = None
            response = client.create(model=f"{provider}/{model}", **kwargs)
        else:
            # Handle unknown provider
            response = "Unknown provider"

        return response

    def embed(
        self, input: List[str], model: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Embeds input text using the specified model and provider.

        Args:
            input (List[str]): List of input texts to be embedded.
            model (Optional[str]): The model to use for embeddings, in the format "provider/model".
            **kwargs (Any): Additional keyword arguments for embedding configuration.

        Returns:
            Dict[str, Any]: A dictionary containing the embeddings response.
        """
        kwargs = {**self.config, **kwargs}
        input = input or kwargs.get("input")
        model = model or kwargs.get("model")
        if model is not None:
            provider, model = model.split("/")
        else:
            provider = "ai21"
            model = "ai21_Embeddings_Model"

        if input is None:
            raise ValueError("Please provide 'input' for embeddings")

        updated_kwargs = update_kwargs_with_config(provider, **kwargs)
        updated_kwargs = remove_unwanted_params(**updated_kwargs)

        if provider in {"anthtropic", "perplexity", "groq"}:
            raise ValueError(
                f"Embeddings are not accessible for the provider: {provider}. Please consider utilizing OpenAI Embeddings instead."
            )

        provider_clients = {
            "ai21": (AI21, ["api_key"]),
            "alephalpha": (ALEPHALPHA, ["token"]),
            "azure": (
                AZURE,
                [
                    "api_key",
                    "azure_ad_token_provider",
                    "deployment_name",
                    "base_url",
                    "api_version",
                    "azure_endpoint",
                ],
            ),
            "cohere": (COHERE, ["api_key"]),
            "ollama": (OLLAMA, []),
            "openai": (OPENAI, ["api_key", "base_url"]),
            "togetherai": (TOGETHERAI, ["api_key"]),
            "sagemaker": (
                SAGEMAKER,
                ["aws_access_key_id", "aws_secret_access_key", "aws_region_name"],
            ),
            "bedrock": (
                BEDROCK,
                [
                    "aws_role_name",
                    "aws_session_name",
                    "aws_access_key_id",
                    "aws_secret_access_key",
                    "aws_region_name",
                    "aws_bedrock_runtime_endpoint",
                    "timeout",
                ],
            ),
            "huggingface": (HUGGINGFACE, ["api_key"]),
        }
        # Check if provider is in the dictionary and get the client instantiation method and required kwargs
        if provider in provider_clients:
            client_class, required_kwargs = provider_clients[provider]
            kwargs_to_pass = {
                key: getattr(self, key) for key in required_kwargs if hasattr(self, key)
            }
            client = client_class(**kwargs_to_pass)
            response = None
            if provider == "ai21":
                response = client.embed(input=input, **kwargs)
            else:
                response = client.embed(
                    input=input, model=f"{provider}/{model}", **kwargs
                )

        return response

    def messages(self, response: dict) -> Any:
        try:
            object_type = response["object"]
            choices = response.get("choices", [])
            tools = []
            for choice in choices:
                if "tool_calls" in choice.get("messages", {}):
                    tools.extend(choice["messages"]["tool_calls"])

                if tools:
                    # Check if tools[0] is not None and if it's a dictionary
                    if (
                        tools[0] is not None
                        and isinstance(tools[0], dict)
                        and "function" in tools[0]
                    ):
                        function_args = tools[0]["function"].get("arguments")
                        return function_args

                    # If tools list is empty or if the first tool is None or not a dictionary, fallback to using choices
                    if choices:
                        message = choices[0].get("messages", {}).get("content", "")
                        return message
                    else:
                        return "No message available"
                elif object_type == "list":
                    message = response["data"][0]["embedding"]
                    return message
                else:
                    return "Unsupported object type"
        except (json.JSONDecodeError, ValueError):
            return "Error parsing response"

    def simulate_streaming(self, message, chunk_size=100):
        for pos in range(0, len(message), chunk_size):
            yield message[pos : pos + chunk_size]
            # Simulate delay for streaming effect
            time.sleep(0.1)

    def cost(self, response) -> Union[str, Any]:
        object_type = response["object"]

        if object_type == "chat.completion":
            model_name = response["model"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
            model_config = configurations.get(model_name)

            if model_config is None:
                print("Model configuration not found for", model_name)
                return None
            input_cost = model_config.input_price_per_token * prompt_tokens
            output_cost = model_config.output_price_per_token * completion_tokens
            total_cost = input_cost + output_cost
            return {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            }
        elif object_type == "list":
            model_name = response["model"]
            prompt_tokens = response.usage.prompt_tokens
            embed_model_config = embed_configurations.get(model_name)

            if embed_model_config is None:
                raise ValueError(
                    "Model configuration or input price per token is missing."
                )
            input_cost = embed_model_config.price_per_token * prompt_tokens
            total_cost = input_cost
            return {
                "input_cost": input_cost,
                "total_cost": total_cost,
            }

    def usage(self, response) -> Union[str, Any]:
        object = response["object"]
        if object == "chat.completion":
            usage = response.get("usage", {})
        else:
            prompt_tokens = response.usage.prompt_tokens
            total_tokens = response.usage.total_tokens

            usage = {
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
            }

        return usage
