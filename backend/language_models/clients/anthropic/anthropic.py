import json
import requests
import boto3
from typing import Any, Dict, List, Optional
from ....base.base import LLM
from ..utilities.utils import (
    update_kwargs_with_config,
    remove_unwanted_params,
    get_headers,
    build_payload,
    build_request_url,
    check_response,
    build_response_model,
)
from anthropic import AnthropicBedrock, AnthropicVertex
from ...prompts.prompts_template import (
    anthropic_messages_prompt,
    prompt_to_messages,
)


class ANTHROPIC(LLM):

    def __init__(
        self,
        api_key: Optional[str] = None,
        aws_access_key: Optional[str] = None,
        aws_secret_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is not None:
            kwargs.update(config)
        self.api_key = api_key
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.aws_session_token = aws_session_token
        self.aws_region = aws_region
        self.project_id = project_id
        self.region = region

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
        tools = kwargs.get("tools")
        stream = kwargs.get("stream")
        if prompt is not None and tools is not None:
            kwargs.pop("prompt")
            messages = prompt_to_messages(prompt)
            kwargs.update({"messages": messages, **kwargs})

        api_key = self.api_key
        aws_access_key = self.aws_access_key
        aws_secret_key = self.aws_secret_key
        aws_session_token = self.aws_session_token
        aws_region = self.aws_region
        region = self.region
        project_id = self.project_id
        provider, model = model.split("/")
        if prompt is not None:

            if (
                (aws_access_key and aws_secret_key and aws_session_token and aws_region)
                is not None
                or model.startswith("anthropic.")
                or (project_id and region) is not None
                or "@" in model
                or model.startswith("claude-3")
            ):
                messages = prompt_to_messages(prompt)
                messages = anthropic_messages_prompt(messages)
                kwargs.update({"messages": messages})
                if "prompt" in kwargs:
                    kwargs.pop("prompt")

        updated_kwargs = update_kwargs_with_config(provider, **kwargs)
        updated_kwargs = remove_unwanted_params(**updated_kwargs)
        headers = get_headers(provider, api_key=api_key)
        payload = build_payload(model, provider, **updated_kwargs)
        url = build_request_url(provider, model, payload)
        if model.startswith("anthropic."):
            if aws_access_key and aws_secret_key and aws_session_token and aws_region:
                client = AnthropicBedrock(
                    aws_access_key=aws_access_key,
                    aws_secret_key=aws_secret_key,
                    aws_session_token=aws_session_token,
                    aws_region=aws_region,
                )
                if stream:
                    response_dict = {}
                    with client.messages.stream(**payload) as stream:
                        for messages in stream.text_stream:
                            print(messages, end="", flush=True)

                        accumulated = stream.get_final_message()
                        response_dict = accumulated.model_dump_json(indent=2)

                else:
                    response = client.messages.create(**payload)
                    response_dict = json.loads(response.content)
            else:
                bedrock = boto3.client(service_name="bedrock-runtime")
                if "model" in payload:
                    payload.pop("model")
                # TODO: streaming response
                response = bedrock.invoke_model(body=payload, modelId=model)
                response_dict = json.loads(response.get("body").read())
        elif "@" in model and project_id and region:
            client = AnthropicVertex(project_id=project_id, region=region)
            response = client.messages.create(**payload)
            response_dict = {"content": [{"text": {response}}]}
        else:
            response = requests.post(url=url, json=payload, headers=headers)
            check_response(provider, response, url)
            if stream:
                final_response, response_stream = stream_response(url, headers, payload)
                response_dict = json.dumps(final_response)
            else:
                response_dict = json.loads(response.text)

        if stream:
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
        else:
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
        pass


def handle_event(event_type, data, response):
    if event_type == "message_start":
        response["message"] = data["message"]
        response["content"] = []
    elif event_type == "content_block_start":
        response["content"].append({"type": data["content_block"]["type"], "text": ""})
    elif event_type == "content_block_delta":
        response["content"][-1]["text"] += data["delta"]["text"]
    elif event_type == "content_block_stop":
        pass  # No action needed
    elif event_type == "message_delta":
        response["stop_reason"] = data["delta"]["stop_reason"]
        response["stop_sequence"] = data["delta"]["stop_sequence"]
        response["usage"] = data["usage"]
    elif event_type == "message_stop":
        pass  # No action needed
    elif event_type == "ping":
        pass  # No action needed
    elif event_type == "error":
        pass  # Handle error event
    else:
        pass  # Handle unknown event type


def stream_response(url, headers, payload):
    response = {
        "message": {},
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": None,
    }

    response_stream = []

    with requests.post(url, headers=headers, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                event_type, data = line.split(b":", 1)
                event_type = event_type.decode("utf-8").strip()
                data = json.loads(data.decode("utf-8"))

                handle_event(event_type, data, response)

                response_stream.append((event_type, data))

    return response, response_stream
