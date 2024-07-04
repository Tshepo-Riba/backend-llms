import copy
import json
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    create_sagemaker_client,
    handle_sagemaker_service_data,
    build_response_model,
    build_embeddings_response,
)


class SAGEMAKER(LLM):

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region_name = aws_region_name

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
        contentType = "application/json"
        updated_kwargs = copy.deepcopy(kwargs)

        if self.aws_access_key_id is not None:
            client = create_sagemaker_client(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region_name=self.aws_region_name,
            )
        else:
            client = create_sagemaker_client(
                aws_region_name=self.aws_region_name,
            )

        payload = handle_sagemaker_service_data(**updated_kwargs)
        response = client.invoke_model(
            body=payload,
            EndpointName=model,
            CustomAttributes="accept_eula=true",
            contentType=contentType,
        )
        response_dict = json.loads(response["Body"].read().decode("utf8"))
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
        if input is None:
            raise ValueError("Please provide 'input' for embeddings")
        contentType = "application/json"

        provider, model = model.split("/")
        if self.aws_access_key_id is not None:
            client = create_sagemaker_client(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_region_name=self.aws_region_name,
            )
        else:
            client = create_sagemaker_client(
                aws_region_name=self.aws_region_name,
            )
        payload = json.dumps({"text_inputs": input}).encode("utf-8")

        print(payload)
        response = client.invoke_model(
            body=payload,
            EndpointName=model,
            CustomAttributes="accept_eula=true",
            contentType=contentType,
        )

        response_dict = json.loads(response["Body"].read().decode("utf8"))

        response_model = build_embeddings_response(
            provider=provider,
            input=input,
            model=model,
            response_dict=response_dict,
        )
        return response_model
