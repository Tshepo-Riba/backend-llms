import json
import requests
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    request_process_response,
    update_kwargs_with_config,
    update_embed_kwargs_with_config,
    remove_unwanted_params,
    build_payload,
    build_request_url,
    check_response,
    build_embed_payload,
    build_embed_url,
    build_embeddings_response,
)


class OLLAMA(LLM):

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)

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

        prompt = kwargs.get("prompt")
        messages = kwargs.get("messages")

        provider, model = model.split("/")

        updated_kwargs = update_kwargs_with_config(provider, **kwargs)
        updated_kwargs = remove_unwanted_params(updated_kwargs)

        payload = build_payload(model, provider, **updated_kwargs)
        url = build_request_url(provider, model, payload)
        print(url)
        print(payload)
        response = requests.post(url=url, json=payload)
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
        if input is None:
            raise ValueError("Please provide 'input' for embeddings")

        provider, model = model.split("/")
        updated_kwargs = update_embed_kwargs_with_config(provider, **kwargs)

        payload = build_embed_payload(provider, input, model, **updated_kwargs)
        url = build_embed_url(provider, model, payload)
        print(url)
        print(payload)
        response = requests.post(url=url, json=payload)
        check_response(provider, response, url)

        response_dict = {}
        if response.text:
            try:
                response_dict = json.loads(response.text)
            except json.JSONDecodeError as e:
                print("Error decoding JSON:", e)
            else:
                try:
                    response_dict = json.loads(response.text)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
        response_model = build_embeddings_response(
            provider=provider,
            input=input,
            model=model,
            response_dict=response_dict,
        )

        return response_model
