import json
import requests
import google.auth
import vertexai
from typing import Any, Dict, List, Optional
from ....base.base import LLM

from ..utilities.utils import (
    update_kwargs_with_config,
    remove_unwanted_params,
    get_headers,
    check_response,
    build_response_model,
    build_embed_payload,
    build_embed_url,
    build_embeddings_response,
    handle_gemini_service_data,
    update_vertex_kwargs_with_config_and_safety_settings,
)

try:
    import google.generativeai as genai
except:
    raise Exception(
        "Failed to import google.generativeai. Please ensure you have installed it by running pip install -q google-generativeai"
    )
try:
    import google.generativeai as palm
except:
    raise Exception(
        "Failed to import google.generativeai. Please ensure you have installed it by running pip install -q google-generativeai"
    )


class GOOGLE(LLM):

    def __init__(
        self,
        api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        config: Optional[dict] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        **kwargs,
    ) -> None:
        # TODO:fix config
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.api_key = api_key
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location
        self.google_api_key = google_api_key

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
        api_key = self.api_key

        provider, model = model.split("/")
        genai.configure(api_key=api_key)
        if provider == "gemini":
            gemini_model = genai.GenerativeModel(f"models/{model}")
            safety_settings, prompt = handle_gemini_service_data(**updated_kwargs)
            response = gemini_model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(**updated_kwargs),
                safety_settings=safety_settings,
            )
            response_dict = {}
            if response.text:
                try:
                    response_dict = json.dumps({"responses": response.text})
                except json.JSONDecodeError as e:
                    print("Error decoding JSON:", e)
                else:
                    try:
                        response_dict = json.dumps({"responses": response.text})
                    except json.JSONDecodeError as e:
                        print("Error decoding JSON:", e)

        else:
            credentials = google.auth.default(quota_project_id=self.vertex_project)[0]
            vertexai.init(
                project=self.vertex_project,
                location=self.vertex_location,
                credentials=credentials,
                **kwargs,
            )
            updated_kwargs, safety_settings = (
                update_vertex_kwargs_with_config_and_safety_settings(**updated_kwargs)
            )
            prompt = " ".join(
                message["content"]
                for message in messages
                if isinstance(message.get("content"), str)
            )
            type_model = ""
            client_options = {
                "api_endpoint": f"{self.vertex_location}-aiplatform.googleapis.com"
            }
            instances = None
            # TODO: model response to vertaix
        if messages:
            updated_kwargs.pop("messages", None)
        else:
            updated_kwargs.pop("prompt", None)

        response_model = build_response_model(
            response_dict=response_dict,
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

        if input is None:
            raise ValueError("Please provide 'input' for embeddings")
        api_key = self.api_key or self.google_api_key

        client = genai.configure(api_key=api_key)
        updated_kwargs = update_kwargs_with_config(provider, kwargs)
        updated_kwargs = remove_unwanted_params(updated_kwargs)

        if provider in {"anthtopic", "perplexity", "groq"}:
            raise ValueError(
                f"Embeddings are not accessible for the provider: {provider}. Please consider utilizing OpenAI Embeddings instead."
            )

        if provider in {"openai"}:
            if kwargs.get("base_url") or kwargs.get("api_base"):
                headers = get_headers(provider, kwargs)
                payload = build_embed_payload(provider, input, **updated_kwargs)

                url = build_embed_url(provider, model, updated_kwargs)
                response = requests.post(url=url, json=payload, headers=headers)

                check_response(provider, response)
                response_dict = json.loads(response.text)
            else:
                response_dict = client.embeddings.create(**updated_kwargs)
        elif provider in {"alephalpha", "ai21", "cohere", "ollama", "togetherai"}:
            headers = get_headers(provider, kwargs)
            payload = build_embed_payload(provider, input, **updated_kwargs)
            url = build_embed_url(provider, model, updated_kwargs)
            response = requests.post(url=url, json=payload, headers=headers)
            check_response(provider, response)
            response_dict = json.loads(response.text)

        response_model = build_embeddings_response(provider, response_dict)

        return response_model
