import copy
import json
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    remove_unwanted_params,
    update_bedrock_kwargs_with_config,
    check_response,
    build_bedrock_response_model,
    create_bedrock_client,
    handle_service_data,
    build_embed_payload,
    build_bedrock_embeddings_response,
)

try:
    import google.generativeai as genai
except ImportError:
    raise Exception(
        "Failed to import google.generativeai. Please ensure you have installed it by running pip install -q google-generativeai"
    )
try:
    import google.generativeai as palm
except ImportError:
    raise Exception(
        "Failed to import google.generativeai. Please ensure you have installed it by running pip install -q google-generativeai"
    )


class BEDROCK(LLM):

    def __init__(
        self,
        config: Optional[dict] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_bedrock_runtime_endpoint: Optional[str] = None,
        aws_session_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_role_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name
        self.aws_bedrock_runtime_endpoint = aws_bedrock_runtime_endpoint
        self.aws_session_name = aws_session_name
        self.aws_profile_name = aws_profile_name
        self.aws_role_name = aws_role_name

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
        service = model.split("/")[1].split(".")[0]
        provider, model = model.split("/")
        timeout = kwargs.get("timeout")
        accept = "application/json"
        contentType = "application/json"

        updated_kwargs = update_bedrock_kwargs_with_config(service, model, **kwargs)
        updated_kwargs = remove_unwanted_params(**updated_kwargs)

        client = updated_kwargs.pop("aws_bedrock_client", None)
        if client is None:
            if self.aws_role_name is not None and self.aws_session_name is not None:
                client = create_bedrock_client(
                    aws_role_name=self.aws_role_name,
                    aws_session_name=self.aws_session_name,
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )
            elif self.aws_access_key_id is not None:
                client = create_bedrock_client(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )
            else:
                client = create_bedrock_client(
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )

        if service == "anthropic":
            payload = handle_service_data(
                service=service,
                updated_kwargs=updated_kwargs,
                model=model,
            )
        else:
            payload = handle_service_data(
                service=service,
                updated_kwargs=updated_kwargs,
            )

        response = client.invoke_model(
            body=payload, modelId=model, accept=accept, contentType=contentType
        )
        check_response(provider, response)
        response_dict = json.loads(response.get("body").read())

        response_model = build_bedrock_response_model(
            response_dict=response_dict,
            prompt=prompt if not messages else None,
            messages=messages if messages else None,
            provider=provider,
            model=model,
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
        service = model.split("/")[1].split(".")[0]

        updated_kwargs = update_bedrock_kwargs_with_config(service, model, **kwargs)
        updated_kwargs = remove_unwanted_params(**updated_kwargs)
        timeout = kwargs.get("timeout")

        client = updated_kwargs.pop("aws_bedrock_client", None)
        if client is None:
            if self.aws_role_name is not None and self.aws_session_name is not None:
                client = create_bedrock_client(
                    aws_role_name=self.aws_role_name,
                    aws_session_name=self.aws_session_name,
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )
            elif self.aws_access_key_id is not None:
                client = create_bedrock_client(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )
            else:
                client = create_bedrock_client(
                    aws_region_name=self.aws_region_name,
                    aws_bedrock_runtime_endpoint=self.aws_bedrock_runtime_endpoint,
                    timeout=timeout,
                )
        payload = build_embed_payload(provider, input, model, **updated_kwargs)
        response_dict = client.embeddings.create(**payload)
        response_model = build_bedrock_embeddings_response(provider, response_dict)

        return response_model
