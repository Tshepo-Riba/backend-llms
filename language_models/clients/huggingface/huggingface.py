import os
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    update_kwargs_with_config,
    remove_unwanted_params,
    build_response_model,
    process_model_and_invoke_inference,
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


class HUGGINGFACE(LLM):

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.api_key = api_key

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
        device = kwargs.get("device")
        task = kwargs.get("task")

        provider, model = model.split("/")

        if provider == "huggingface":
            file_path = os.path.join(
                os.path.dirname(__file__),
                "_metadata",
                "huggingface",
                "models_info_text_gen.txt",
            )
            if os.path.exists(file_path):
                # Read the file with comma delimiter
                with open(file_path, "r", encoding="utf-8") as file:
                    for line in file:
                        columns = line.strip().split(",")
                        if len(columns) > 0 and model in columns[0]:
                            model = columns[0]

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

        if messages:
            if task:
                response_dict = process_model_and_invoke_inference(
                    device=device,
                    model=model,
                    task=task,
                    messages=messages,
                )
            elif updated_kwargs.get("method"):
                if updated_kwargs.get("method") == "constrained_beam_search":
                    response_dict = process_model_and_invoke_inference(
                        device=device,
                        model=model,
                        method=updated_kwargs.get("method"),
                        messages=messages,
                        constraints=updated_kwargs.get("constraints"),
                    )
                else:
                    response_dict = process_model_and_invoke_inference(
                        device=device,
                        model=model,
                        messages=messages,
                        method=updated_kwargs.get("method"),
                    )
        else:
            if updated_kwargs.get("method"):
                if updated_kwargs.get("method") == "constrained_beam_search":
                    response_dict = process_model_and_invoke_inference(
                        device=device,
                        model=model,
                        method=updated_kwargs.get("method"),
                        prompt=prompt,
                        constraints=updated_kwargs.get("constraints"),
                    )
                else:
                    response_dict = process_model_and_invoke_inference(
                        device=device,
                        model=model,
                        prompt=prompt,
                        method=updated_kwargs.get("method"),
                    )
            else:
                response_dict = process_model_and_invoke_inference(
                    device=device,
                    model=model,
                    prompt=prompt,
                )

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
        pass
