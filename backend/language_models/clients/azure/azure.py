import json
import os
import tiktoken
import requests
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    request_process_response,
    update_kwargs_with_config,
    remove_unwanted_params,
    build_payload,
    get_headers,
    build_embed_payload,
    build_embeddings_response,
    azure_build_request_url,
    check_response,
)

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from ...prompts.prompts_template import prompt_to_messages


class AZURE(LLM):

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_ad_token_provider: Optional[str] = None,
        deployment_name: Optional[str] = None,
        resource_name: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.base_url = base_url
        self.api_key = api_key
        self.api_version = api_version
        self.azure_endpoint = azure_endpoint
        self.azure_ad_token_provider = azure_ad_token_provider
        self.deployment_name = deployment_name
        self.resource_name = resource_name

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
        api_key = self.api_key
        prompt = kwargs.get("prompt")
        messages = kwargs.get("messages")
        stream = kwargs.get("stream")

        tools = kwargs.get("tools")
        if prompt is not None and tools is not None:
            kwargs.pop("prompt")
            messages = prompt_to_messages(prompt)
            kwargs.update({"messages": messages, **kwargs})

        azure_credentials = {
            "resource_name": self.resource_name,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
        }

        provider, model = model.split("/")
        if model.startswith("gpt-3.5"):
            model = model.replace("gpt-3.5", "gpt-35")

        if not model:
            raise ValueError("Please ensure that a model is provided.")

        if (messages is not None and prompt is not None) or (
            messages is None and prompt is None
        ):
            raise ValueError(
                "Please provide either a 'message' or a 'prompt', but not both."
            )

        updated_kwargs = update_kwargs_with_config(provider, kwargs)
        updated_kwargs = remove_unwanted_params(updated_kwargs)
        headers = get_headers(provider, api_key=api_key)

        payload = build_payload(model, provider, **updated_kwargs)
        url = azure_build_request_url(provider, model, payload, azure_credentials)
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            stream=stream if stream is not None else False,
        )

        check_response(provider, response, url)

        if messages:
            updated_kwargs.pop("messages", None)
        else:
            updated_kwargs.pop("prompt", None)

        response_model = request_process_response(
            response,
            prompt=prompt if not messages else None,
            messages=messages,
            provider=provider,
            model=model,
            **updated_kwargs,
        )

        return response_model

    def embed(
        self, input: List[str], model: Optional[str], **kwargs: Any
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
        if model is None:
            raise ValueError("Please provide a 'model' for embeddings")

        provider, model = model.split("/")
        azure_credentials = {
            "resource_name": self.resource_name,
            "deployment_name": self.deployment_name,
            "api_version": self.api_version,
            "azure_endpoint": self.azure_endpoint,
        }

        if input is None:
            raise ValueError("Please provide 'input' for embeddings")
        api_key = self.api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = self.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = self.api_version or os.getenv("AZURE_API_VERSION")
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            self.azure_ad_token_provider or os.getenv("AZURE_AD_TOKEN_PROVIDER"),
        )

        updated_kwargs = update_kwargs_with_config(provider, kwargs)
        updated_kwargs = remove_unwanted_params(updated_kwargs)
        print(updated_kwargs)
        if self.azure_ad_token_provider is not None:
            client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_ad_token_provider=token_provider,
            )
        else:
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=azure_endpoint,
            )
        payload = build_embed_payload(provider, input, model, **updated_kwargs)
        response_dict = client.embeddings.create(**payload)
        response_model = build_embeddings_response(provider, response_dict)

        return response_model
