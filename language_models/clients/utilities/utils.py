import base64
import csv
import importlib
import os
import re
import json
import time
import traceback
import requests
import tiktoken
import boto3
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from typing import Any, Literal, Optional, Union, Iterator, Dict, Generator, List, Tuple
from botocore.client import BaseClient

from ...responses.utilities import generate_prefixed_hex_id, get_finish_reason
from ...responses.chat.chat_response import ChatResponse, Choice
from ...responses.embedding.embedding_response import EmbeddingsResponse
from ...responses.image.image_response import ImageResponse
from ...responses.transcript.transcript_reponse import TranscriptResponse
from ...responses.streaming.streaming_response import (
    Choice as DeltaChoice,
    Delta,
    StreamingResponse,
)
from ...responses.types.types import ChatResponseMessage
from ...exceptions.exceptions import OpenAIException

from ...configurations.configs import (
    BedrockConfigFactory,
    AI21Configuration,
    AlephAlphaConfiguration,
    AnthropicConfiguration,
    AzureOpenAIConfiguration,
    CohereConfiguration,
    HuggingfaceConfiguration,
    OllamaConfiguration,
    OpenAIConfiguration,
    TogetherAIConfiguration,
    SagemakerConfiguration,
    PalmConfiguration,
    GeminiConfiguration,
    VertexConfiguration,
    GroqConfiguration,
    PerplexityConfiguration,
    AmazonBedrockAI21Configuration,
    AmazonBedrockAnthropicConfiguration,
    AmazonBedrockAnthropicClaude3Configuration,
    AmazonBedrockCohereConfiguration,
    AmazonBedrockLlamaConfiguration,
    AmazonBedrockMistralConfiguration,
    AmazonBedrockStabilityConfiguration,
    AmazonBedrockTitanConfiguration,
    AlephAlphaEmbeddingsConfiguration,
    CohereEmbeddingConfiguration,
)
from ...exceptions.exceptions import (
    ExceptionFactory,
)
from ...prompts.prompts_template import (
    ai21_message_prompt,
    alephalpha_message_prompt,
    anthropic_prompt,
    function_call_prompt,
    anthropic_messages_prompt,
    cohere_message_prompt,
    generate_custom_prompt,
    message_to_prompt,
    prompt_to_messages,
)
from ...responses.types.types import (
    EmbeddingsUsage,
)

from ...responses.embedding.embedding_response import EmbeddingsResponse


from transformers import (
    LogitsProcessorList,
    StoppingCriteriaList,
    BeamScorer,
    MinLengthLogitsProcessor,
    MaxLengthCriteria,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    HammingDiversityLogitsProcessor,
    ConstrainedBeamSearchScorer,
    PhrasalConstraint,
    PreTrainedModel,
    pipeline,
    PreTrainedTokenizer,
    BatchEncoding,
)

from optimum.pipelines import pipeline as opt_pipeline

from google.cloud.aiplatform_v1beta1.types import content as gapic_content_types


from vertexai.preview.generative_models import (
    Part,
)


embedding_models: List = ["text-embedding-ada-002"]


def update_function_call(
    function_call_chunk: Any,
    full_function_call: Dict[str, Any],
    completion_tokens: int,
) -> Tuple[Dict[str, Any], int]:
    """
    Update the function call from the chunk.

    Args:
    - function_call_chunk (Any): The function call chunk.
    - full_function_call (Dict[str, Any]): The full function call.
    - completion_tokens (int): The number of completion tokens.

    Returns:
    - Tuple[Dict[str, Any], int]: The updated full function call and the updated number of completion tokens.
    """
    # Handle function call
    if function_call_chunk:
        if full_function_call is None:
            full_function_call = {}
        for field in ["name", "arguments"]:
            completion_tokens += update_dict_from_chunk(
                function_call_chunk, full_function_call, field
            )

    if full_function_call:
        return full_function_call, completion_tokens
    else:
        raise RuntimeError("Function call is not found, this should not happen.")


def update_tool_calls(
    tool_calls_chunk: Any,
    full_tool_call: Dict[str, Any],
    completion_tokens: int,
) -> Tuple[Dict[str, Any], int]:
    """Update the function call from the chunk.

    Args:
        function_call_chunk: The function call chunk.
        full_function_call: The full function call.
        completion_tokens: The number of completion tokens.

    Returns:
        The updated full function call and the updated number of completion tokens.

    """
    if tool_calls_chunk.type and tool_calls_chunk.type != "function":
        raise NotImplementedError(
            f"Tool call type {tool_calls_chunk.type} is currently not supported. "
            "Only function calls are supported."
        )

    # Handle tool call
    assert full_tool_call is None or isinstance(full_tool_call, dict), full_tool_call
    if tool_calls_chunk:
        if full_tool_call is None:
            full_tool_call = {}
        for field in ["index", "id", "type"]:
            completion_tokens += update_dict_from_chunk(
                tool_calls_chunk, full_tool_call, field
            )

        if hasattr(tool_calls_chunk, "function") and tool_calls_chunk.function:
            if "function" not in full_tool_call:
                full_tool_call["function"] = None

            full_tool_call["function"], completion_tokens = update_function_call(
                tool_calls_chunk.function,
                full_tool_call["function"],
                completion_tokens,
            )

    if full_tool_call:
        return full_tool_call, completion_tokens
    else:
        raise RuntimeError("Tool call is not found, this should not happen.")


def update_dict_from_chunk(chunk: Any, d: Dict[str, Any], field: str) -> int:
    """
    Update the dict from the chunk.

    Reads `chunk.field` and if present updates `d[field]` accordingly.

    Args:
    - chunk: The chunk.
    - d: The dict to be updated in place.
    - field: The field.

    Returns:
    - int: The number of completion tokens added.
    """
    assert isinstance(d, dict), d
    new_value = getattr(chunk, field, None)
    completion_tokens = 0
    if new_value is not None:
        if isinstance(new_value, (list, dict)):
            raise NotImplementedError(
                f"Field {field} is a list or dict, which is currently not supported. "
                "Only strings and numbers are supported."
            )
        d[field] = d.get(field, "") + str(new_value)
        completion_tokens = 1
    return completion_tokens


############################################################################ Client Utilities #########################################################################################
def convert_to_streaming_response(
    response_object: Optional[Dict[str, Any]] = None
) -> Generator[StreamingResponse, None, None]:
    """
    Converts response object to streaming response.

    Args:
        response_object (dict, optional): The response object to convert. Defaults to None.

    Yields:
        StreamingResponse: Streaming response object.

    Raises:
        Exception: Raised when response_object is None.
    """
    if response_object is None:
        raise Exception("Error in response object format")

    model_response_object = StreamingResponse()
    choice_list = []
    for idx, choice in enumerate(response_object["choices"]):
        delta = Delta(
            content=choice["message"].get("content", None),
            role=choice["message"]["role"],
            tools=choice["message"].get("tools", None),
        )
        finish_reason = choice.get("finish_reason", None)
        if finish_reason is None:
            finish_reason = choice.get("finish_details")
        logprobs = choice.get("logprobs", None)
        choice = DeltaChoice(
            finish_reason=finish_reason,
            index=idx,
            delta=delta,
            logprobs=logprobs,
        )

        choice_list.append(choice)
    model_response_object.choices = choice_list

    if "usage" in response_object and response_object["usage"] is not None:
        model_response_object.usage.completion_tokens = response_object["usage"].get("completion_tokens", 0)  # type: ignore
        model_response_object.usage.prompt_tokens = response_object["usage"].get("prompt_tokens", 0)  # type: ignore
        model_response_object.usage.total_tokens = response_object["usage"].get("total_tokens", 0)  # type: ignore

    if "id" in response_object:
        model_response_object.id = response_object["id"]

    if "created" in response_object:
        model_response_object.created = response_object["created"]

    if "system_fingerprint" in response_object:
        model_response_object.system_fingerprint = response_object["system_fingerprint"]

    if "model" in response_object:
        model_response_object.model = response_object["model"]
    yield model_response_object


def model_response_object(
    response_object: Optional[dict] = None,
    model_response_object: Optional[
        Union[
            ChatResponse,
            EmbeddingsResponse,
            ImageResponse,
            TranscriptResponse,
            StreamingResponse,
        ]
    ] = None,
    response_type: Literal[
        "completion",
        "embedding",
        "image_generation",
        "audio_transcription",
        "streaming",
    ] = "completion",
    stream: bool = False,
) -> Union[
    ChatResponse,
    EmbeddingsResponse,
    ImageResponse,
    TranscriptResponse,
    StreamingResponse,
]:
    """
    Model response object conversion function.

    Args:
        response_object (dict, optional): The response object to convert. Defaults to None.
        model_response_object (Union[ChatResponse, StreamingResponse, EmbeddingsResponse, ImageResponse, TranscriptResponse], optional): The response object type. Defaults to None.
        response_type (Literal["completion", "embedding", "image_generation", "audio_transcription"], optional): The type of response object. Defaults to "completion".
        stream (bool, optional): Whether to use streaming response. Defaults to False.

    Returns:
        Union[ChatResponse, EmbeddingsResponse, ImageResponse, TranscriptResponse]: The converted response object.

    Raises:
        Exception: Raised when encountering errors in response object format.
    """
    try:
        if response_type == "completion" and (
            model_response_object is None
            or isinstance(model_response_object, ChatResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            choice_list = []
            for idx, choice in enumerate(response_object["choices"]):
                message = ChatResponseMessage(
                    content=choice["message"].get("content", None),
                    role=choice["message"]["role"],
                    functions=choice["message"].get("functions", None),
                    tools=choice["message"].get("tools", None),
                )
                finish_reason = choice.get("finish_reason", None)
                if finish_reason is None:
                    finish_reason = choice.get("finish_details")
                logprobs = choice.get("logprobs", None)
                choice = Choice(
                    finish_reason=finish_reason,
                    index=idx,
                    message=message,
                    logprobs=logprobs,
                )
                choice_list.append(choice)
            model_response_object.choices = choice_list

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object[
                    "usage"
                ].get("completion_tokens", 0)
                model_response_object.usage.prompt_tokens = response_object[
                    "usage"
                ].get("prompt_tokens", 0)
                model_response_object.usage.total_tokens = response_object["usage"].get(
                    "total_tokens", 0
                )

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "id" in response_object:
                model_response_object.id = response_object["id"]

            if "system_fingerprint" in response_object:
                model_response_object.system_fingerprint = response_object[
                    "system_fingerprint"
                ]

            return model_response_object

        if response_type == "streaming" and (
            model_response_object is None
            or isinstance(model_response_object, StreamingResponse)
        ):
            if response_object is None or model_response_object is None:
                raise Exception("Error in response object format")
            if stream:
                return convert_to_streaming_response(response_object=response_object)

        elif response_type == "embedding" and (
            model_response_object is None
            or isinstance(model_response_object, EmbeddingsResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = EmbeddingsResponse()

            if "model" in response_object:
                model_response_object.model = response_object["model"]

            if "object" in response_object:
                model_response_object.object = response_object["object"]

            model_response_object.data = response_object["data"]

            if "usage" in response_object and response_object["usage"] is not None:
                model_response_object.usage.completion_tokens = response_object[
                    "usage"
                ].get("completion_tokens", 0)
                model_response_object.usage.prompt_tokens = response_object[
                    "usage"
                ].get("prompt_tokens", 0)
                model_response_object.usage.total_tokens = response_object["usage"].get(
                    "total_tokens", 0
                )

            return model_response_object
        elif response_type == "image_generation" and (
            model_response_object is None
            or isinstance(model_response_object, ImageResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = ImageResponse()

            if "created" in response_object:
                model_response_object.created = response_object["created"]

            if "data" in response_object:
                model_response_object.data = response_object["data"]

            return model_response_object
        elif response_type == "audio_transcription" and (
            model_response_object is None
            or isinstance(model_response_object, TranscriptResponse)
        ):
            if response_object is None:
                raise Exception("Error in response object format")

            if model_response_object is None:
                model_response_object = TranscriptResponse()

            if "text" in response_object:
                model_response_object.text = response_object["text"]

    except Exception as e:
        raise Exception(f"Invalid response object {traceback.format_exc()}")


def num_tokens_from_messages(messages, model):
    num_tokens = 0
    encoding = tiktoken.encoding_for_model(model)
    for message in messages:
        for key, value in message.items():
            if key == "content" and isinstance(value, str):
                # Check if the value is a string
                num_tokens += len(encoding.encode(value))
                # Assuming `encoding` is an instance of the encoder
                # Adjust this line based on your specific implementation
    return num_tokens


def setup_hearders(**kwargs):
    api_key = kwargs.get("api_key")
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class Streamer:
    SPECIAL_TOKENS = ["<|assistant|>", "<|system|>", "<|user|>", "<s>", "</s>"]

    def __init__(
        self,
        completion_stream: Iterator[Any],
        model: Any,
        provider: Optional[Any] = None,
    ):
        self.model = model
        self.provider = provider
        self.completion_stream = completion_stream
        self.sent_first_chunk = False
        self.sent_last_chunk = False
        self.holding_chunk = ""
        self.response_id = None
        self.created_at = None

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def __next__(self):
        if self.sent_last_chunk:
            raise StopIteration
        chunk = next(self.completion_stream)
        return self.chunk_creator(chunk)

    def handle_chunk_error(self, chunk: str):
        raise ValueError(f"Unable to parse response. Original response: {chunk}")

    def decode_chunk(self, chunk: bytes) -> Dict[str, Any]:
        chunk = chunk.decode("utf-8")
        if chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])
            return {
                "text": data_json.get("token", {}).get("text", ""),
                "is_finished": bool(
                    data_json.get("details", {}).get("finish_reason", "")
                ),
                "finish_reason": data_json.get("details", {}).get("finish_reason", ""),
            }
        elif "error" in chunk:
            self.handle_chunk_error(chunk)
        return {"text": "", "is_finished": False, "finish_reason": ""}

    def handle_azure_chunk(self, chunk: str) -> Dict[str, Any]:
        if "data: [DONE]" in chunk:
            return {"text": "", "is_finished": True, "finish_reason": "stop"}
        elif chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])
            choices = data_json.get("choices", [])
            text = choices[0]["delta"].get("content", "") if choices else ""
            finish_reason = choices[0].get("finish_reason", "") if choices else ""
            return {
                "text": text,
                "is_finished": bool(finish_reason),
                "finish_reason": finish_reason,
            }
        elif "error" in chunk:
            self.handle_chunk_error(chunk)
        return {"text": "", "is_finished": False, "finish_reason": ""}

    def handle_openai_text_completion_chunk(self, chunk: str) -> Dict[str, Any]:
        if "data: [DONE]" in chunk or self.sent_last_chunk:
            raise StopIteration
        elif chunk.startswith("data:"):
            data_json = json.loads(chunk[5:])
            choices = data_json.get("choices", [])
            text = choices[0].get("text", "") if choices else ""
            finish_reason = choices[0].get("finish_reason", "") if choices else ""
            self.sent_last_chunk = bool(finish_reason)
            return {
                "text": text,
                "is_finished": bool(finish_reason),
                "finish_reason": finish_reason,
            }
        elif "error" in chunk:
            self.handle_chunk_error(chunk)
        return {"text": "", "is_finished": False, "finish_reason": None}

    def handle_openai_chat_completion_chunk(self, chunk: Any) -> Dict[str, Any]:
        if chunk.choices:
            choice = chunk.choices[0]
            text = choice.delta.content if choice.delta else ""
            finish_reason = choice.finish_reason if choice.finish_reason else None
            return {
                "text": text,
                "is_finished": bool(finish_reason),
                "finish_reason": finish_reason,
                "original_chunk": chunk,
            }
        return {"text": "", "is_finished": False, "finish_reason": None}

    def handle_palm_chunk(self) -> Dict[str, Any]:
        response_obj = {}
        if len(self.completion_stream) == 0:
            if self.sent_last_chunk:
                raise StopIteration
            else:
                self.sent_last_chunk = True
                response_obj["text"] = ""
                response_obj["is_finished"] = True
                response_obj["finish_reason"] = "stop"
        else:
            chunk_size = 30
            new_chunk = "".join(
                [next(self.completion_stream) for _ in range(chunk_size)]
            )
            response_obj["text"] = new_chunk
            time.sleep(0.05)
            response_obj["is_finished"] = False
            response_obj["finish_reason"] = None
        return response_obj

    def handle_ollama_chat_stream(self, chunk: Any) -> Dict[str, Any]:
        if isinstance(chunk, dict):
            json_chunk = chunk
        else:
            json_chunk = json.loads(chunk)
        if json_chunk.get("error"):
            raise Exception(f"Ollama Error - {json_chunk}")

        if json_chunk.get("done"):
            return {"text": "", "is_finished": True, "finish_reason": "stop"}

    def handle_cached_response_chunk(self, chunk: Any) -> Dict[str, Any]:
        response_obj = {
            "text": chunk.choices[0].delta.content,
            "is_finished": True,
            "finish_reason": chunk.choices[0].finish_reason,
            "original_chunk": chunk,
        }
        return response_obj

    def handle_anthropic_chunk(self, chunk: bytes) -> Dict[str, Any]:
        str_line = chunk.decode("utf-8")
        if str_line.startswith("data:"):
            data_json = json.loads(str_line[5:])
            if data_json.get("type") == "message_delta":
                return {
                    "text": "",
                    "is_finished": True,
                    "finish_reason": data_json.get("delta", {}).get("stop_reason"),
                }
            elif data_json.get("type") == "content_block_delta":
                return {
                    "text": data_json.get("delta", {}).get("text", ""),
                    "is_finished": False,
                    "finish_reason": None,
                }
        elif "error" in str_line:
            self.handle_chunk_error(
                f"Unable to parse response. Original response: {str_line}"
            )
        return {"text": "", "is_finished": False, "finish_reason": ""}

    def chunk_creator(self, chunk):
        model_response = StreamingResponse(stream=True, model=self.model)
        if self.response_id is not None:
            model_response.id = self.response_id
        else:
            self.response_id = model_response.id
        model_response.provider = {"provider": self.provider}
        model_response.created_at = {"created_at": time.time()}
        model_response.choices = [DeltaChoice()]
        model_response.choices[0].finish_reason = None
        response_obj = {}
        try:
            completion_obj = {"content": ""}
            if self.provider == "anthropic":
                response_obj = self.handle_anthropic_chunk(chunk)
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    model_response.choices[0].finish_reason = response_obj[
                        "finish_reason"
                    ]
            else:
                response_obj = self.handle_openai_chat_completion_chunk(chunk)
                if response_obj is None:
                    return
                completion_obj["content"] = response_obj["text"]
                if response_obj["is_finished"]:
                    if response_obj["finish_reason"] == "error":
                        raise OpenAIException(
                            "The Mistral API encountered a streaming error, indicating that no content string was provided along with the error finish reason."
                        )
                    model_response.choices[0].finish_reason = response_obj[
                        "finish_reason"
                    ]
                if (
                    "original_chunk" in response_obj
                    and response_obj["original_chunk"] is not None
                ):
                    model_response.system_fingerprint = getattr(
                        response_obj["original_chunk"], "system_fingerprint", None
                    )
                    if hasattr(response_obj["original_chunk"], "id"):
                        model_response.id = response_obj["original_chunk"].id
                    if len(response_obj["original_chunk"].choices) > 0:
                        try:
                            delta = dict(
                                response_obj["original_chunk"].choices[0].delta
                            )
                            model_response.choices[0].delta = Delta(**delta)
                        except Exception:
                            model_response.choices[0].delta = Delta()
                    if self.sent_first_chunk == False:
                        model_response.choices[0].delta.role = "assistant"
                        self.sent_first_chunk = True
                if "logprobs" in response_obj and response_obj["logprobs"] is not None:
                    model_response.choices[0].logprobs = response_obj["logprobs"]

            if (
                "content" in completion_obj
                and isinstance(completion_obj["content"], str)
                and len(completion_obj["content"]) > 0
            ):
                hold, model_response_str = self.check_special_tokens(
                    chunk=completion_obj["content"],
                    finish_reason=model_response.choices[0].finish_reason,
                )
                if not hold:
                    original_chunk = response_obj.get("original_chunk", None)
                    if original_chunk:
                        model_response.id = original_chunk.id
                        if len(original_chunk.choices) > 0:
                            try:
                                delta = dict(original_chunk.choices[0].delta)
                                model_response.choices[0].delta = Delta(**delta)
                            except Exception:
                                model_response.choices[0].delta = Delta()
                        model_response.system_fingerprint = (
                            original_chunk.system_fingerprint
                        )
                        if self.sent_first_chunk == False:
                            model_response.choices[0].delta.role = "assistant"
                            self.sent_first_chunk = True
                    else:
                        completion_obj["content"] = model_response_str
                        if self.sent_first_chunk == False:
                            completion_obj["role"] = "assistant"
                            self.sent_first_chunk = True
                        model_response.choices[0].delta = Delta(**completion_obj)
                    return model_response
                else:
                    return
            elif model_response.choices[0].finish_reason is not None:
                if len(self.holding_chunk) > 0:
                    if model_response.choices[0].delta.content is None:
                        model_response.choices[0].delta.content = self.holding_chunk
                    else:
                        model_response.choices[0].delta.content += self.holding_chunk
                    self.holding_chunk = ""
                model_response.choices[0].finish_reason = get_finish_reason(
                    model_response.choices[0].finish_reason
                )
                return model_response
            elif (
                getattr(model_response.choices[0].delta, "tool_calls", None) is not None
                or getattr(model_response.choices[0].delta, "function_call", None)
                is not None
            ):
                if self.sent_first_chunk == False:
                    model_response.choices[0].delta.role = "assistant"
                    self.sent_first_chunk = True
                return model_response
            else:
                return
        except StopIteration:
            raise StopIteration
        except Exception as e:
            traceback.print_exc()
            e.message = str(e)
            raise OpenAIException(
                model=self.model,
                provider=self.provider,
                original_exception=e,
            )

    def check_special_tokens(self, chunk: str, finish_reason: Optional[str]):
        hold = False
        if finish_reason:
            for token in self.SPECIAL_TOKENS:
                if token in chunk:
                    chunk = chunk.replace(token, "")
            return hold, chunk

        if self.sent_first_chunk is True:
            return hold, chunk

        curr_chunk = self.holding_chunk + chunk
        curr_chunk = curr_chunk.strip()

        for token in self.SPECIAL_TOKENS:
            if len(curr_chunk) < len(token) and curr_chunk in token:
                hold = True
            elif len(curr_chunk) >= len(token):
                if token in curr_chunk:
                    self.holding_chunk = curr_chunk.replace(token, "")
                    hold = True
            else:
                pass

        if hold is False:  # reset
            self.holding_chunk = ""
        return hold, curr_chunk


def update_kwargs_with_config(provider: str, **kwargs: Dict) -> Dict:
    """Update kwargs with configuration values based on the provider.

    Args:
        provider (str): The name of the provider.
        kwargs (Dict): Keyword arguments to be updated with configuration values.

    Returns:
        Dict: Updated keyword arguments.
    """
    PROVIDER_CONFIG_MAPPING = {
        "ai21": AI21Configuration,
        "alephalpha": AlephAlphaConfiguration,
        "anthropic": AnthropicConfiguration,
        "azure": AzureOpenAIConfiguration,
        "cohere": CohereConfiguration,
        "huggingface": HuggingfaceConfiguration,
        "ollama": OllamaConfiguration,
        "openai": OpenAIConfiguration,
        "groq": GroqConfiguration,
        "togetherai": TogetherAIConfiguration,
        "sagemaker": SagemakerConfiguration,
        "palm": PalmConfiguration,
        "gemini": GeminiConfiguration,
        "vertex": VertexConfiguration,
        "perplexity": PerplexityConfiguration,
    }

    configuration_class = PROVIDER_CONFIG_MAPPING.get(provider)
    if configuration_class is not None:
        if provider == "bedrock":
            # Get configuration using BedrockConfigFactory
            configuration = BedrockConfigFactory.get_provider(provider)
        else:
            # Creating an instance of configuration_class before calling config()
            configuration_instance = configuration_class()
            configuration = configuration_instance.config()

        # Update kwargs with missing values
        for k, v in configuration.items():
            kwargs.setdefault(k, v)

        if provider == "anthropic":
            # Check if "prompt" or "messages" is in kwargs
            if "prompt" in kwargs:
                # Convert "max_tokens" to "max_tokens_to_sample" if provider is "anthropic"
                if "max_tokens" in kwargs:
                    kwargs["max_tokens_to_sample"] = kwargs.pop("max_tokens")
            elif "messages" in kwargs:
                # Convert "max_tokens" to "max_tokens_to_sample" if provider is "anthropic"
                if "max_tokens" in kwargs:
                    del kwargs["max_tokens_to_sample"]

    else:
        print(f"No configuration found for provider '{provider}'")

    return kwargs


def update_bedrock_kwargs_with_config(service: str, model: str, **kwargs: Dict) -> Dict:
    """Update kwargs with configuration values based on the provider.

    Args:
        provider (str): The name of the provider.
        kwargs (Dict): Keyword arguments to be updated with configuration values.

    Returns:
        Dict: Updated keyword arguments.
    """

    PROVIDER_CONFIG_MAPPING = {
        "ai21": AmazonBedrockAI21Configuration,
        "anthropic": (
            AmazonBedrockAnthropicClaude3Configuration
            if model.startswith("claude-3")
            else AmazonBedrockAnthropicConfiguration
        ),
        "amazon": AmazonBedrockTitanConfiguration,
        "cohere": AmazonBedrockCohereConfiguration,
        "meta": AmazonBedrockLlamaConfiguration,
        "mistral": AmazonBedrockMistralConfiguration,
        "stability": AmazonBedrockStabilityConfiguration,
    }

    configuration_class = PROVIDER_CONFIG_MAPPING.get(service)
    if configuration_class is not None:
        configuration_instance = configuration_class()
        configuration = configuration_instance.config()

        # Update kwargs with missing values
        for k, v in configuration.items():
            kwargs.setdefault(k, v)

        if service == "anthropic":
            # Check if "prompt" or "messages" is in kwargs
            if "prompt" in kwargs:
                # Convert "max_tokens" to "max_tokens_to_sample" if provider is "anthropic"
                if "max_tokens" in kwargs:
                    kwargs["max_tokens_to_sample"] = kwargs.pop("max_tokens")
            elif "messages" in kwargs:
                # Convert "max_tokens" to "max_tokens_to_sample" if provider is "anthropic"
                if "max_tokens" in kwargs:
                    del kwargs["max_tokens_to_sample"]

    else:
        print(f"No configuration found for provider '{service}'")

    return kwargs


def update_embed_kwargs_with_config(provider: str, **kwargs: Dict) -> Dict:
    """Update kwargs with configuration values based on the provider.

    Args:
        provider (str): The name of the provider.
        **kwargs (Dict): Keyword arguments to be updated with configuration values.

    Returns:
        Dict: Updated keyword arguments.
    """
    PROVIDER_CONFIG_MAPPING = {
        "ai21": AI21Configuration,
        "alephalpha": AlephAlphaEmbeddingsConfiguration,
        "anthropic": AnthropicConfiguration,
        "azure": OpenAIConfiguration,
        "cohere": CohereEmbeddingConfiguration,
        "huggingface": HuggingfaceConfiguration,
        "ollama": OllamaConfiguration,
        "openai": OpenAIConfiguration,
        "togetherai": TogetherAIConfiguration,
        "sagemaker": SagemakerConfiguration,
        "palm": PalmConfiguration,
        "gemini": GeminiConfiguration,
        "vertex": VertexConfiguration,
    }

    configuration_class = PROVIDER_CONFIG_MAPPING.get(provider)
    if configuration_class is not None:
        if provider == "bedrock":
            # Get configuration using BedrockConfigFactory
            configuration = BedrockConfigFactory.get_provider(provider)
        else:
            # Creating an instance of configuration_class before calling config()
            configuration_instance = configuration_class()
            configuration = configuration_instance.config()

        # Update kwargs with missing values
        for k, v in configuration.items():
            kwargs.setdefault(k, v)
    else:
        print(f"No configuration found for provider '{provider}'")

    return kwargs


def remove_unwanted_params(**kwargs: Dict) -> Dict:
    """Remove unwanted parameters from kwargs.

    Args:
        kwargs (Dict): Keyword arguments to be filtered.

    Returns:
        Dict: Filtered keyword arguments.
    """

    unwanted_params = [
        "api_base",
        "headers",
        "base_url",
        "api_version",
        "azure_endpoint",
        "api_key",
    ]

    # Check if kwargs is a dictionary
    if isinstance(kwargs, dict):
        return {k: v for k, v in kwargs.items() if k not in unwanted_params}
    else:
        raise ValueError("Input kwargs must be a dictionary.")


def build_default_headers(
    provider: str,
    token: Optional[str] = None,
    api_key: Optional[str] = None,
    agent: Optional[str] = None,
) -> Dict[str, str]:
    """Build default headers based on the provider, token, and API key.

    Args:
        provider (str): The name of the provider.
        token (str): Token to be included in the header.
        api_key (str): API key to be included in the header.
        **kwargs (Dict[str, str]): Other keyword arguments.

    Returns:
        Dict[str, str]: Default headers.
    """
    if provider and "alephalpha" in provider.lower():
        bearer = token
    else:
        bearer = api_key

    if provider == "anthropic":
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": str(bearer),
        }
    else:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": "Bearer " + str(bearer),
        }

    return headers


def get_headers(
    provider: str,
    token: Optional[str] = None,
    api_key: Optional[str] = None,
    agent: Optional[str] = None,
    **kwargs: Dict[str, str],
) -> Dict[str, str]:
    """Get headers.

    Args:
        **kwargs (Dict[str, str]): Keyword arguments.

    Returns:
        Dict[str, str]: Headers.
    """
    return kwargs.get("headers") or build_default_headers(
        provider, token, api_key, agent
    )


def build_payload(model: str, provider: str, **updated_kwargs) -> Dict[str, Any]:
    """Builds a payload based on the provider, model, and updated keyword arguments.

    Args:
       updated_kwargs (Dict[str, Any]): Updated keyword arguments.

    Returns:
        Dict[str, Any]: The constructed payload.
    """
    prompt = updated_kwargs.pop("prompt", None)
    messages = updated_kwargs.pop("messages", None)
    task = updated_kwargs.pop("task", None)

    if task is not None and provider != "huggingface":
        raise ValueError("This task is exclusive to the 'huggingface' provider.")

    tools = updated_kwargs.pop("tools", None)
    if tools is not None:
        messages = function_call_prompt(messages, tools)

    payload = {}

    if messages:
        if provider == "ai21":
            converted_messages, system_message = ai21_message_prompt(messages)
            payload.update(
                {
                    "messages": converted_messages,
                    "system": system_message,
                    **updated_kwargs,
                }
            )
        elif provider == "alephalpha":
            prompt = alephalpha_message_prompt(messages)
            payload.update({"prompt": prompt, **updated_kwargs})

        elif provider in {
            "cohere",
            "anthropic",
            "openai",
            "azure",
            "togetherai",
            "groq",
            "perplexity",
        }:
            if provider == "cohere":
                # TODO: include Chat history for Cohere
                messages = cohere_message_prompt(messages)
                payload.update({"message": messages, **updated_kwargs})
            else:
                if provider == "anthropic":
                    messages = anthropic_messages_prompt(messages)
                payload.update({"messages": messages, **updated_kwargs})
        elif provider == "ollama":
            payload.update(
                {
                    "messages": messages,
                    "options": {**updated_kwargs},
                }
            )
    elif prompt:
        if provider in {
            "anthropic",
            "openai",
            "azure",
            "cohere",
            "alephalpha",
            "togetherai",
            "ai21",
            "groq",
            "perplexity",
        }:
            if provider == "anthropic":
                prompt = anthropic_prompt(prompt)
                payload.update({"prompt": prompt, **updated_kwargs})
            elif provider in ("openai", "azure", "groq", "perplexity", "togetherai"):
                messages = prompt_to_messages(prompt)
                payload.update({"messages": messages, **updated_kwargs})
            else:
                payload.update({"prompt": prompt, **updated_kwargs})
        elif provider == "ollama":
            payload.update(
                {
                    "prompt": prompt,
                    "options": {**updated_kwargs},
                }
            )

    if provider != "ai21":
        # TODO: Include model strings for TogetherAI  and Huggingface from files or database
        payload["model"] = model

    return payload


def build_embed_payload(
    provider: str,
    input_data: Union[str, list],
    model: str,
    **updated_kwargs,
) -> Dict[str, Any]:
    payload = {}

    if provider in {"ai21"}:
        payload["texts"] = input_data
        payload["type"] = "segment"
    elif provider == "cohere":
        payload["texts"] = input_data
    elif provider == "alephalpha":
        if isinstance(input_data, list):
            input_data = ". ".join(input_data)  # Convert list to string
        payload.update(
            {
                "prompt": input_data,
            }
        )
    elif provider in {
        "ollama",
    }:
        if provider == "ollama":
            if isinstance(input_data, list):
                input_data = ". ".join(input_data)  # Convert list to string
                payload.update(
                    {
                        "prompt": input_data,
                    }
                )
        else:
            payload.update(
                {
                    "prompt": input_data,
                }
            )
    elif provider in ("azure", "openai", "togetherai"):
        if isinstance(input_data, list):
            input_data = ". ".join(input_data)  # Convert list to string
        payload.update(
            {
                "input": input_data,
            }
        )
    else:
        payload.update(
            {
                "input": input_data,
            }
        )

    payload.update(**updated_kwargs)

    if provider != "ai21":
        payload["model"] = model
    return payload


def build_request_url(
    provider: str,
    model: Optional[str],
    kwargs: Dict[str, Any],
) -> str:
    """Builds a request URL based on the provider, model, and keyword arguments.

    Args:
        provider (str): The name of the provider.
        model (Optional[str]): The model to be used.
        kwargs (Dict[str, Any]): Keyword arguments.

    Returns:
        str: The constructed request URL.

    Raises:
        ValueError: If the API base URL or model is missing, or if the provider is not supported.
    """

    SERVICE_URLS = {
        "ai21": "https://api.ai21.com/studio/v1/",
        "alephalpha": "https://api.aleph-alpha.com",
        "anthropic": "https://api.anthropic.com/v1",
        "ollama": "http://localhost:11434/api",
        "perplexity": "https://api.perplexity.ai",
        "cohere": "https://api.cohere.ai/v1",
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "togetherai": "https://api.together.xyz",
        # TODO:Huggingface Rest API
    }

    api_base = kwargs.get("api_base", SERVICE_URLS.get(provider))

    if not api_base:
        raise ValueError(f"No API base URL found for provider '{provider}'.")

    if provider == "ai21":
        if kwargs.get("prompt") is not None:
            endpoint = "complete"
        else:
            endpoint = "chat"
        if not model:
            raise ValueError(f"Model is required for provider '{provider}'.")
        return f"{api_base}{model}/{endpoint}"

    endpoint_map = {
        "alephalpha": "complete",
        "anthropic": "complete" if kwargs.get("prompt") is not None else "messages",
        "ollama": "generate" if kwargs.get("prompt") is not None else "chat",
        "cohere": "generate" if kwargs.get("prompt") is not None else "chat",
        "perplexity": "chat/completions",
        "groq": "chat/completions",
        "openai": (
            "completions" if kwargs.get("prompt") is not None else "chat/completions"
        ),
        "togetherai": (
            "completions" if kwargs.get("prompt") is not None else "chat/completions"
        ),
        # "huggingface":"" TODO: Huggingface Rest API
    }

    if provider in endpoint_map:
        return f"{api_base}/{endpoint_map[provider]}"

    raise ValueError(f"Provider '{provider}' not supported.")


def azure_build_request_url(
    provider: str,
    model: Optional[str],
    kwargs: Dict[str, Any],
    azure_credentials: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a request URL based on the provider, model, and keyword arguments.

    Args:
        provider (str): The name of the provider.
        model (Optional[str]): The model to be used.
        kwargs (Dict[str, Any]): Keyword arguments.

    Returns:
        str: The constructed request URL.

    Raises:
        ValueError: If the API base URL or model is missing, or if the provider is not supported.
    """
    AZURE_RESOURCE_NAME = azure_credentials.get("resource_name")
    AZURE_DEPLOYMENT_NAME = azure_credentials.get("deployment_name")
    AZURE_API_VERSION = azure_credentials.get("api_version")

    SERVICE_URLS = {
        "azure": f"https://{AZURE_RESOURCE_NAME}.openai.azure.com/openai/deployments/{AZURE_DEPLOYMENT_NAME}",
    }

    api_base = kwargs.get("api_base", SERVICE_URLS.get(provider))

    if not api_base:
        raise ValueError(f"No API base URL found for provider '{provider}'.")

    if provider == "ai21":
        if kwargs.get("prompt") is not None:
            endpoint = "complete"
        else:
            endpoint = "chat"
        if not model:
            raise ValueError(f"Model is required for provider '{provider}'.")
        return f"{api_base}{model}/{endpoint}"

    endpoint_map = {
        "azure": (
            f"/completions?{AZURE_API_VERSION}"
            if kwargs.get("prompt") is not None
            else f"/chat/completions?{AZURE_API_VERSION}"
        ),
    }

    if provider in endpoint_map:
        return f"{api_base}/{endpoint_map[provider]}"

    raise ValueError(f"Provider '{provider}' not supported.")


def build_embed_url(provider: str, model: Optional[str], kwargs: Dict[str, Any]) -> str:
    """Builds a request URL based on the provider, model, and keyword arguments.

    Args:
        provider (str): The name of the provider.
        model (Optional[str]): The model to be used.
        kwargs (Dict[str, Any]): Keyword arguments.

    Returns:
        str: The constructed request URL.

    Raises:
        ValueError: If the API base URL or model is missing, or if the provider is not supported.
    """
    SERVICE_URLS = {
        "ai21": "https://api.ai21.com/studio/v1",
        "alephalpha": "https://api.aleph-alpha.com",
        "anthropic": "https://api.anthropic.com/v1",
        "ollama": "http://localhost:11434/api",
        "perplexity": "https://api.perplexity.ai",
        "cohere": "https://api.cohere.ai/v1",
        "openai": "https://api.openai.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "togetherai": "https://api.together.xyz/v1",
        # TODO:Huggingface Rest API
    }
    api_base = kwargs.get("api_base", SERVICE_URLS.get(provider))

    endpoint_map = {
        "ai21": "embed",
        "alephalpha": "embed",
        "ollama": "embeddings",
        "cohere": "embed",
        "openai": "embeddings",
        "togetherai": "embeddings",
        # "huggingface":"" TODO: Huggingface Rest API
    }

    if provider in endpoint_map:
        return f"{api_base}/{endpoint_map[provider]}"

    raise ValueError(f"Provider '{provider}' not supported.")


def check_response(
    provider: str, response: requests.Response, url: Optional[str]
) -> None:
    """Checks the response for errors and raises an appropriate exception if necessary.

    Args:
        provider (str): The name of the provider.
        response (requests.Response): The response object to check.
        url (Optional[str]): The URL associated with the request.

    Raises:
        Exception: An exception specific to the provider with status code and message.
    """
    if response.status_code != 200:
        exception_class = ExceptionFactory.get_exception(
            provider.lower()
        )  # Ensure lowercase
        raise exception_class(response.status_code, response.text, url)


def build_response_model(
    response_dict: Dict[str, Any],
    messages: Optional[List[str]] = None,
    prompt: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Builds a response model based on the provided data.

    Args:
        response_dict (Dict[str, Any]): The dictionary containing the response data.
        messages (Optional[List[str]]): The list of messages (default: None).
        prompt (Optional[str]): The prompt string (default: None).
        provider (Optional[str]): The provider string (default: None).

    Returns:
        Dict[str, Any]: The constructed response model.
    """
    stream = kwargs.get("stream")
    encoding = tiktoken.get_encoding("cl100k_base")
    provider_handlers = {
        "ai21": handle_ai21,
        "alephalpha": handle_alephalpha,
        "anthropic": handle_streaming_anthropic if stream else handle_anthropic,
        "ollama": handle_ollama,
        "cohere": handle_cohere,
        "perplexity": handle_perplexity_provider,
        "togetherai": handle_togetherai_provider,
        "openai": handle_openai_provider,
        "azure": handle_openai_provider,
        "groq": handle_openai_provider,
        "hugginface": handle_hugginface,
    }
    if provider == "cohere" and messages is not None:
        (
            content,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            logprobs,
            chat_history,
        ) = provider_handlers.get(provider)(response_dict, messages, prompt, encoding)
    else:
        content, finish_reason, prompt_tokens, completion_tokens, logprobs = (
            provider_handlers.get(provider)(response_dict, messages, prompt, encoding)
        )
    stream = kwargs.get("stream")
    tools = kwargs.get("tools")
    if tools is not None:
        id = generate_prefixed_hex_id("call_", 9)
        type = tools[0]["type"]
        function_name = tools[0]["function"]["name"]
        if provider == "cohere" or (provider == "groq" and model == "gemma-7b-it"):
            arguments = content.replace("```json\n", "").replace("\n```", "")
        elif provider == "groq" and (
            model.startswith("mixtral") or model.startswith("mistral")
        ):
            arguments_json = json.loads(content)
            arguments = arguments_json["parameters"]
        else:
            arguments = content

        content = None

        tools = {
            "id": id,
            "type": type,
            "function": {"name": function_name, "arguments": arguments},
        }
        finish_reason = "tools_calls"

    if provider == "cohere" and messages is not None:
        response_model = {
            "id": generate_prefixed_hex_id("chatcmpl-", 12),
            "model": model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "messages": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tools],
                    },
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                }
            ],
            "chat_history": chat_history,
            "system_fingerprint": generate_prefixed_hex_id("fp_", 9),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    else:

        response_model = {
            "id": generate_prefixed_hex_id("chatcmpl-", 12),
            "model": model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "messages": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tools],
                    },
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                }
            ],
            "system_fingerprint": generate_prefixed_hex_id("fp_", 9),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return response_model


def build_embeddings_usage(input: Any) -> EmbeddingsUsage:
    """
    Builds embeddings usage based on the input.

    Args:
        input (Any): The input data for which embeddings usage needs to be computed.

    Returns:
        EmbeddingsUsage: An instance of EmbeddingsUsage with prompt_tokens and total_tokens set.

    Raises:
        None
    """
    # Initialize tokenizer
    encoding = tiktoken.get_encoding("cl100k_base")

    # Calculate prompt tokens
    prompt_tokens = len(encoding.encode(str(input)))

    # Create and return EmbeddingsUsage object
    return EmbeddingsUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)


def build_embeddings_response(
    provider: Any, model: Any, input: Optional[str], response_dict: Dict[str, Any]
) -> None:
    """
    Builds embeddings response based on the provider, model, input text, and response dictionary.

    Args:
        provider (Any): The provider of the embeddings.
        model (Any): The model used for generating embeddings.
        input (str): The input text for which embeddings response is being built.
        response_dict (Dict[str, Any]): A dictionary containing response information.

    Returns:
        None

    Raises:
        None
    """
    embeddings_data = []
    usage = None

    if provider in {"openai", "togetherai", "ai21", "ollama", "azure"}:
        embeddings = None
        if provider in {"openai", "togetherai"}:
            embeddings = response_dict["data"][0]["embedding"]
            index = response_dict["data"][0]["index"]
            model = response_dict.get("model", model)
        elif provider == "ai21":
            embeddings = response_dict["results"][0]["embedding"]
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": embeddings,
                }
            )
            usage = build_embeddings_usage(input)
        elif provider == "ollama":
            embeddings = response_dict["embedding"]
        if provider != "ai21":
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embeddings,
                }
            )

            if provider in {"openai", "togetherai", "azure"}:
                usage = build_embeddings_usage(input)

    elif provider == "cohere":
        embeddings = response_dict["embeddings"]
        for index, embedding in enumerate(embeddings):
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": embedding,
                }
            )
        usage = {
            "prompt_tokens": response_dict["meta"]["billed_units"]["input_tokens"],
            "total_tokens": response_dict["meta"]["billed_units"]["input_tokens"],
        }

    elif provider == "alephalpha":
        possible_values = {
            "mean": "mean",
            "weighted_mean": "weighted_mean",
            "max": "max",
            "last_token": "last_token",
            "abs_max": "abs_max",
        }
        for layer_name, layer_data in response_dict["embeddings"].items():
            if layer_name.startswith("layer"):
                layer_index = int(layer_name.split("_")[-1])
                for aggregation_method, values in layer_data.items():
                    if aggregation_method in possible_values:
                        embeddings_data.append(
                            {
                                "object": "embedding",
                                "index": layer_index,
                                "embedding": values,
                                "aggregation_method": possible_values[
                                    aggregation_method
                                ],
                            }
                        )
        usage = {
            "prompt_tokens": response_dict["num_tokens_prompt_total"],
            "total_tokens": response_dict["num_tokens_prompt_total"],
        }
    elif provider == "bedrock":
        embeddings_data = []
        embeddings = response_dict["embeddings"]
        for idx, embedding in enumerate(embeddings):
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding,
                }
            )

    return EmbeddingsResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=usage,
    )


def build_bedrock_response_model(
    response_dict: Dict[str, Any],
    messages: Optional[List[str]] = None,
    prompt: Optional[str] = None,
    service: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Builds a response model based on the provided data.

    Args:
        response_dict (Dict[str, Any]): The dictionary containing the response data.
        messages (Optional[List[str]]): The list of messages (default: None).
        prompt (Optional[str]): The prompt string (default: None).
        provider (Optional[str]): The provider string (default: None).

    Returns:
        Dict[str, Any]: The constructed response model.
    """

    encoding = tiktoken.get_encoding("cl100k_base")
    service_handlers = {
        "ai21": handle_ai21,
        "anthropic": handle_anthropic,
        "amazon": handle_amazon,
        "cohere": handle_cohere,
        "meta": handle_meta,
        "mistral": handle_mistral,
    }
    if service == "cohere" and messages is not None:
        (
            content,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            logprobs,
            chat_history,
        ) = service_handlers.get(service)(response_dict, messages, prompt, encoding)
    else:
        content, finish_reason, prompt_tokens, completion_tokens, logprobs = (
            service_handlers.get(service)(response_dict, messages, prompt, encoding)
        )

    tools = kwargs.get("tools")
    if tools is not None:
        for tool in tools:
            id = generate_prefixed_hex_id("call_", 9)
            type = tool["type"]
            function_name = tool["function"]["name"]
            arguments = tool.get("content", None)

            tool_data = {
                "id": id,
                "type": type,
                "function": {"name": function_name, "arguments": arguments},
            }
            finish_reason = "tools_calls"

    if service == "cohere" and messages is not None:
        response_model = {
            "id": generate_prefixed_hex_id("chatcmpl-", 12),
            "model": model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "messages": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tools],
                    },
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                }
            ],
            "chat_history": chat_history,
            "system_fingerprint": generate_prefixed_hex_id("fp_", 9),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    else:

        response_model = {
            "id": generate_prefixed_hex_id("chatcmpl-", 12),
            "model": model,
            "created": int(time.time()),
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "messages": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": [tools],
                    },
                    "finish_reason": finish_reason,
                    "logprobs": logprobs,
                }
            ],
            "system_fingerprint": generate_prefixed_hex_id("fp_", 9),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return response_model


def build_bedrock_embeddings_response(
    service: Any, model: Any, input: Optional[str], response_dict: Dict[str, Any]
) -> None:
    """
    Builds embeddings response based on the provider, model, input text, and response dictionary.

    Args:
        provider (Any): The provider of the embeddings.
        model (Any): The model used for generating embeddings.
        input (str): The input text for which embeddings response is being built.
        response_dict (Dict[str, Any]): A dictionary containing response information.

    Returns:
        None

    Raises:
        None
    """
    embeddings_data = []
    usage = None
    encoding = tiktoken.get_encoding("cl100k_base")
    usage = build_embeddings_usage(input)
    if service == "cohere":
        embeddings = response_dict["embeddings"]
        embeddings = [item for sublist in embeddings for item in sublist]

        for idx, embedding in enumerate(embeddings):
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding,
                }
            )
    else:
        embeddings = response_dict["embedding"]
        for idx, embedding in enumerate(embeddings):
            embeddings_data.append(
                {
                    "object": "embedding",
                    "index": idx,
                    "embedding": embedding,
                }
            )

    return EmbeddingsResponse(
        object="list",
        data=embeddings_data,
        model=model,
        usage=usage,
    )


def handle_ai21(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles AI21 API response.

    Args:
        response_dict (Dict[str, Any]): The response from the AI21 API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    if prompt is not None:
        content = response_dict["completions"][0]["data"]["text"]
        finish_reason = response_dict["completions"][0]["finishReason"]["reason"]
        prompt_tokens = len(encoding.encode(str(prompt)))
        completion_tokens = len(encoding.encode(str(content)))
    else:
        content = response_dict["outputs"][0]["text"]
        finish_reason = "stop"
        prompt_tokens = len(
            encoding.encode(str([message["content"] for message in messages]))
        )
        completion_tokens = len(encoding.encode(str(content)))

    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_alephalpha(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles AlephAlpha API response.

    Args:
        response_dict (Dict[str, Any]): The response from the AlephAlpha API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = response_dict["completions"][0]["completion"]
    finish_reason = response_dict["completions"][0]["finish_reason"]
    prompt_tokens = response_dict["num_tokens_prompt_total"]
    completion_tokens = response_dict["num_tokens_generated"]
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_anthropic(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Anthropic API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Anthropic API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    if prompt is not None:
        content = response_dict["completion"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = len(encoding.encode(str(prompt)))
        completion_tokens = len(encoding.encode(str(content)))
    else:
        content = response_dict["content"][0]["text"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = response_dict["usage"]["input_tokens"]
        completion_tokens = response_dict["usage"]["output_tokens"]
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_streaming_anthropic(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Anthropic API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Anthropic API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    if prompt is not None:
        content = response_dict["content"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = len(encoding.encode(str(prompt)))
        completion_tokens = len(encoding.encode(str(content)))
    else:
        content = response_dict["content"][0]["text"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = response_dict["message"]["usage"]["input_tokens"]
        completion_tokens = response_dict["usage"]["output_tokens"]
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_bedrock(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Bedrock API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Anthropic API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    if prompt is not None:
        content = response_dict["completion"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = len(encoding.encode(str(prompt)))
        completion_tokens = len(encoding.encode(str(content)))
    else:
        content = response_dict["content"][0]["text"]
        finish_reason = response_dict["stop_reason"]
        prompt_tokens = response_dict["usage"]["input_tokens"]
        completion_tokens = response_dict["usage"]["output_tokens"]
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_ollama(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Ollama API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Ollama API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = (
        response_dict["message"]["content"] if messages else response_dict["response"]
    )
    finish_reason = "done"
    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = (
        len(encoding.encode(response_dict.get(response_dict["message"]["content"])))
        if messages
        else len(encoding.encode(str(response_dict["response"])))
    )
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_cohere(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Cohere API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Cohere API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    if prompt is not None:
        content = response_dict["generations"][0]["text"]
        finish_reason = response_dict["generations"][0]["finish_reason"]
        prompt_tokens = response_dict["meta"]["billed_units"]["input_tokens"]
        completion_tokens = response_dict["meta"]["billed_units"]["output_tokens"]
        logprobs = None
        return content, finish_reason, prompt_tokens, completion_tokens, logprobs
    else:
        chat_history = response_dict["chat_history"]
        content = response_dict["text"]
        finish_reason = "stop"
        prompt_tokens = len(encoding.encode(str(messages)))
        completion_tokens = len(encoding.encode(str(response_dict["text"])))
        logprobs = None
        return (
            content,
            finish_reason,
            prompt_tokens,
            completion_tokens,
            logprobs,
            chat_history,
        )


def handle_openai_provider(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles OpenAI API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = response_dict["choices"][0]["message"]["content"]
    finish_reason = response_dict["choices"][0]["finish_reason"] if not None else "stop"
    prompt_tokens = response_dict["usage"]["prompt_tokens"]
    completion_tokens = response_dict["usage"]["completion_tokens"]
    logprobs = (
        response_dict["choices"][0]["logprobs"]
        if "logprobs" in response_dict["choices"][0]
        else None
    )
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_perplexity_provider(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles OpenAI API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = response_dict["choices"][0]["message"]["content"]
    finish_reason = "stop"
    prompt_tokens = response_dict["usage"]["prompt_tokens"]
    completion_tokens = response_dict["usage"]["completion_tokens"]
    logprobs = (
        response_dict["choices"][0]["logprobs"]
        if "logprobs" in response_dict["choices"][0]
        else None
    )
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_togetherai_provider(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles OpenAI API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = response_dict["choices"][0]["message"]["content"]
    finish_reason = "stop"
    prompt_tokens = response_dict["usage"]["prompt_tokens"]
    completion_tokens = response_dict["usage"]["completion_tokens"]
    logprobs = (
        response_dict["choices"][0]["logprobs"]
        if "logprobs" in response_dict["choices"][0]
        else None
    )
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_hugginface(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Hugging Face API response.

    Args:
        response_dict (Dict[str, Any]): The response from the Hugging Face API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """
    content = response_dict["responses"]
    finish_reason = "stop"
    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = len(encoding.encode(content))
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_amazon(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Amazon Titan API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """

    content = response_dict["results"][0]["outputText"]
    finish_reason = "stop"
    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = len(encoding.encode(content))
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_meta(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Meta API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """

    content = response_dict["generation"]
    finish_reason = "stop"
    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = len(encoding.encode(content))
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


def handle_mistral(
    response_dict: Dict[str, Any],
    messages: List[Dict[str, Any]],
    prompt: Any,
    encoding: Any,
) -> Tuple[str, str, int, int, None]:
    """
    Handles Meta API response.

    Args:
        response_dict (Dict[str, Any]): The response from the OpenAI API.
        messages (List[Dict[str, Any]]): List of messages.
        prompt (Any): The prompt provided for generation.
        encoding (Any): Encoding object for encoding strings.

    Returns:
        Tuple[str, str, int, int, None]: A tuple containing content, finish reason, prompt tokens,
        completion tokens, and logprobs.
    """

    content = response_dict["outputs"][0]["text"]
    finish_reason = response_dict["outputs"][0]["stop_reason"]
    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = len(encoding.encode(content))
    logprobs = None
    return content, finish_reason, prompt_tokens, completion_tokens, logprobs


################################################# HUGGINGFACE ###############################################################


def fetch_hf_models_api_and_save_to_files() -> None:
    """
    Fetches models data from the Hugging Face API and saves it to a file.

    Raises:
        Exception: If an error occurs during the process.
    """
    try:
        # Make a GET request to the API endpoint
        response = requests.get(
            "https://huggingface.co/api/models",
            headers={"Authorization": "Bearer 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"},
        )

        if response.status_code == 200:
            # Parse the JSON response
            models_data: Dict[str, Any] = response.json()

            # Specify the directory to save files
            save_dir = os.path.join(os.path.dirname(__file__), "meta", "huggingface")

            # Generate file path
            file_path = os.path.join(save_dir, "all_models_data.txt")

            # Check if file already exists
            if not os.path.exists(file_path):
                # Write the data to a CSV file
                with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = set(
                        field for model in models_data for field in model.keys()
                    )
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(models_data)

                print(f"Data has been successfully written to '{file_path}'")
            else:
                print(f"File '{file_path}' already exists. Skipping creation.")
        else:
            print(
                "Failed to fetch data from the API. Status code:", response.status_code
            )

    except Exception as e:
        print("An error occurred:", e)


def create_hf_pipeline(
    task: str,
    device: Optional[str],
    checkpoint: Optional[Union[str, object]] = None,
    tokenizer: Optional[Union[str, object]] = None,
    **kwargs,
) -> pipeline:
    """
    Reference: https://huggingface.co/transformers/v3.1.0/main_classes/pipelines.html, https://huggingface.co/docs/optimum/main/en/onnxruntime/usage_guides/pipelines

    Creates a pipeline object based on the specified task.

    Parameters:
        task (str): The task defining which pipeline will be returned.
        model (str or PreTrainedModel or TFPreTrainedModel, optional): The model to use for the pipeline.
        tokenizer (str or PreTrainedTokenizer, optional): The tokenizer to use for the model.
        framework (str, optional): The framework to use, either "pt" for PyTorch or "tf" for TensorFlow.
        **kwargs: Additional keyword arguments passed to the specific pipeline init.

    Returns:
        Pipeline: A suitable pipeline for the task.
        {
            "generated_text": "string"
        }
    """
    supported_tasks = {
        "feature-extraction",
        "sentiment-analysis",
        "ner",
        "question-answering",
        "fill-mask",
        "summarization",
        "translation_",
        "text-generation",
        "conversation",
        "text-classification",
        "zero-shot-classification",
    }
    optimum_supported_tasks = {
        "feature-extraction",
        "text-classification",
        "token-classification",
        "question-answering",
        "zero-shot-classification",
        "text-generation",
        "text2text-generation",
        "summarization",
        "translation",
        "image-classification",
        "automatic-speech-recognition",
        "image-to-text",
    }

    if task not in supported_tasks:
        raise ValueError(f"Unsupported task: {task}")

    if task == "text-classification":
        task = "sentiment-analysis"

    pipeline_args = {
        "task": task,
        "model": checkpoint,
        "tokenizer": tokenizer,
        **kwargs,
    }
    if task not in optimum_supported_tasks:
        raise ValueError(f"Unsupported task for optimum pipeline: {task}")

    optimum_pipeline_args = {
        "task": task,
        "model": checkpoint,
        "tokenizer": tokenizer,
        "accelerator": "ort",
        **kwargs,
    }

    if device == "cuda":
        optimum_pipeline_args["device"] = device
        return {"responses": opt_pipeline(**optimum_pipeline_args)}
    else:
        return {"responses": pipeline(**pipeline_args)}


def pipeline_huggingface_inference(
    prompt: Optional[str],
    messages: Optional[List[str]],
    model: Union[str, object],
    device: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Process a list of messages using a specified model and additional keyword arguments.

    Parameters:
        prompt (str, optional): A prompt to process.
        messages (List[str], optional): A list of messages to process.
        model (str or PreTrainedModel or TFPreTrainedModel): The model to use for processing.
        **kwargs: Additional keyword arguments to pass to the pipeline.

    Returns:
        Dict[str, Any]: A dictionary containing the processed response(s).
    """
    task = kwargs.get("task")
    tokenizer = kwargs.get("tokenizer")
    responses = []

    if not prompt and not messages:
        return {"responses": responses}

    if messages:
        prompt = message_to_prompt(messages)

    pipe_kwargs = {"model": model, "tokenizer": tokenizer}
    if device:
        pipe_kwargs["device"] = device

    pipe = create_hf_pipeline(task, **pipe_kwargs, **kwargs)
    response = pipe(prompt, **kwargs)
    responses.append(response)

    return {"responses": responses}


def get_auto_model(
    **kwargs: Any,
) -> Tuple[PreTrainedTokenizer, PreTrainedModel]:
    """
    Instantiates a tokenizer and model using Hugging Face's AutoTokenizer and AutoModel.

    Args:
        **kwargs: Additional keyword arguments.
            model_type (str): The type of model to be used.
            tokenizer (str): The name of the tokenizer.
            checkpoint (str): The checkpoint name or path.
            model_args: Additional arguments for model instantiation.

    Returns:
        Tuple[PreTrainedTokenizer, PreTrainedModel]: A tuple containing the tokenizer and model instances.

    Raises:
        TypeError: If any of the arguments is of incorrect type.
        ValueError: If any of the required arguments is not provided.
    """
    model_type = kwargs.get("model_type")
    tokenizer_name = kwargs.get("tokenizer")
    checkpoint = kwargs.get("checkpoint")
    model_args = kwargs.get("model_args")

    if not isinstance(model_type, str):
        raise TypeError("model_type must be a string.")

    if not model_type:
        raise ValueError("model_type must be provided.")

    if not isinstance(tokenizer_name, str):
        raise TypeError("tokenizer must be a string.")

    if not tokenizer_name:
        raise ValueError("tokenizer must be provided.")

    if not isinstance(checkpoint, str):
        raise TypeError("checkpoint must be a string.")

    if not checkpoint:
        raise ValueError("checkpoint must be provided.")

    try:
        # Import the transformers module dynamically
        transformers = importlib.import_module("transformers")

        # Import AutoTokenizer and AutoModel from the dynamically imported module
        AutoTokenizer = getattr(transformers, "AutoTokenizer")
        AutoModel = getattr(transformers, "AutoModel")

    except ImportError:
        # Handle the case where transformers library is not installed
        print("Please make sure you have the transformers library installed.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)

    if model_args:
        model = AutoModel.from_pretrained(checkpoint, *model_args)
    else:
        model = AutoModel.from_pretrained(checkpoint)

    return tokenizer, model


def generate_chat_input(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    tokenize: bool = True,
    add_generation_prompt: bool = True,
) -> Union[BatchEncoding, str]:
    """
    Generate tokenized chat input or formatted chat string based on the provided messages and tokenizer.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer object to be used for tokenization.
        messages (List[Dict[str, str]]): A list of dictionaries representing messages, where each dictionary
            contains 'role' and 'content' keys corresponding to the role of the speaker ('user', 'assistant', etc.)
            and the content of the message respectively.
        tokenize (bool, optional): Whether to tokenize the conversation. Defaults to True.
        add_generation_prompt (bool, optional): Whether to add a generation prompt. Defaults to True.

    Returns:
        Union[BatchEncoding, str]: If tokenize=True, returns a BatchEncoding containing the tokenized input suitable for model input.
            If tokenize=False, returns a formatted string representing the chat conversation.
    """
    template_result = tokenizer.apply_chat_template(messages, tokenize=tokenize)

    if add_generation_prompt:
        generation_prompt = [{"role": "assistant", "content": ""}]
        messages.extend(generation_prompt)

    if tokenize:
        return tokenizer.encode(template_result, return_tensors="pt")
    else:
        return template_result


def generate_sequences(
    checkpoint: Any,
    input_ids: Any,
    device: str,
    method: Union[str, None] = None,
    constraints: Union[None, List[Any]] = None,
    **kwargs: Dict[str, Any],
) -> Any:
    """
    Generate sequences using various decoding methods.

    Args:
        model: The model used for sequence generation.
        input_ids: The input token IDs.
        method: The decoding method to use (default is None).
        constraints: Constraints to apply during decoding (default is None).
        **kwargs: Additional keyword arguments for the generation process.

    Returns:
        The generated sequences.
    """

    if method is None or method == "greedy_search":
        greedy_kwargs = {
            "logits_processor": LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(
                        10, eos_token_id=checkpoint.config.eos_token_id
                    )
                ]
            ),
            "stopping_criteria": StoppingCriteriaList(
                [MaxLengthCriteria(max_length=20)]
            ),
        }
        kwargs.update(greedy_kwargs)
    elif method in ["beam_search", "beam_sample"]:
        beam_kwargs: Dict[str, Any] = {
            "beam_scorer": BeamScorer(
                batch_size=1, num_beams=3, device=checkpoint.to(device)
            ),
            "logits_processor": LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(
                        15, eos_token_id=checkpoint.config.eos_token_id
                    )
                ]
            ),
        }
        if method == "beam_sample":
            beam_kwargs["logits_warper"] = LogitsProcessorList(
                [TopKLogitsWarper(50), TemperatureLogitsWarper(0.7)]
            )
        kwargs.update(beam_kwargs)
    elif method == "group_beam_search":
        kwargs.update(
            {
                "beam_scorer": BeamScorer(
                    batch_size=1,
                    max_length=checkpoint.config.max_length,
                    num_beams=3,
                    device=checkpoint.to(device),
                ),
                "logits_processor": LogitsProcessorList(
                    [
                        HammingDiversityLogitsProcessor(
                            5.5, num_beams=6, num_beam_groups=3
                        ),
                        MinLengthLogitsProcessor(
                            5, eos_token_id=checkpoint.config.eos_token_id
                        ),
                    ]
                ),
            }
        )
    elif method == "constrained_beam_search":
        if constraints is None:
            raise ValueError(
                "Constraints must be provided for constrained_beam_search."
            )
        if isinstance(constraints, list):
            constraints.append(PhrasalConstraint())
        else:
            constraints = [constraints, PhrasalConstraint()]
        kwargs.update(
            {
                "constrained_beam_scorer": ConstrainedBeamSearchScorer(
                    batch_size=1,
                    num_beams=3,
                    device=checkpoint.to(device),
                    constraints=constraints,
                ),
                "logits_processor": LogitsProcessorList(
                    [
                        MinLengthLogitsProcessor(
                            5, eos_token_id=checkpoint.config.eos_token_id
                        )
                    ]
                ),
                "stopping_criteria": StoppingCriteriaList(
                    [MaxLengthCriteria(max_length=20)]
                ),
            }
        )

    kwargs.update(kwargs)

    outputs = checkpoint.generate(input_ids, **kwargs)
    return outputs


def generate_hf_response(
    checkpoint: PreTrainedModel,
    device: Optional[str],
    **updated_kwargs: Dict[str, Any],
) -> Dict[str, str]:
    """
    Generate a response given a checkpoint, prompt, and optional messages.

    Args:
        checkpoint: The pretrained model checkpoint.
        tokenizer: The tokenizer corresponding to the model.
        prompt: The prompt for generating the response.
        messages: Historical messages to include in the conversation.
        generation_prompt: Whether to include a generation prompt.
        method: The decoding method to use (default is None).
        constraints: Constraints to apply during decoding (default is None).
        max_new_tokens: Maximum number of new tokens to generate (default is 50).
        **updated_kwargs: Additional keyword arguments for generation process.

    Returns:
        The generated response.
    """

    generation_prompt = updated_kwargs.get("generation_prompt")
    tokenizer = updated_kwargs.get("tokenizer")
    max_new_tokens = updated_kwargs.get("max_new_tokens")
    method = updated_kwargs.get("method")
    constraints = updated_kwargs.get("constraints")
    prompt = updated_kwargs.get("prompt")
    messages = updated_kwargs.get("messages")

    if messages:
        if isinstance(messages, str):
            messages = [messages]
        if generation_prompt:
            add_generation_prompt = True
        else:
            add_generation_prompt = False

        if device == "cuda":
            input_ids = generate_chat_input(
                tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            ).to(device)
        else:
            input_ids = generate_chat_input(
                tokenizer,
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        kwargs = {
            "input_ids": input_ids,
            "model": checkpoint,
            "max_new_tokens": max_new_tokens,
            **updated_kwargs,
        }
        if method is not None:
            if method == "constrained_beam_search":
                kwargs["constraints"] = constraints
            kwargs["method"] = method
        outputs = generate_sequences(**kwargs)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt", **updated_kwargs)
        outputs = checkpoint.generate(**input_ids)

    return {"responses": tokenizer.decode(outputs[0])}


def process_model_and_invoke_inference(device=None, **kwargs):
    kwargs["checkpoint"] = model
    tokenizer, model = get_auto_model(**kwargs)
    task = kwargs.get("task")
    device = "cuda" if device is None else "cpu"
    messages = kwargs.get("messages")

    if task:
        filtered_kwargs = {
            "task": task,
            "tokenizer": tokenizer,
            "messages": kwargs.get("messages"),
            "prompt": kwargs.get("prompt") if not messages else None,
        }
        filtered_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in filtered_kwargs}
        )
        response = pipeline_huggingface_inference(
            model,
            device,
            **{k: v for k, v in filtered_kwargs.items() if v is not None},
        )
    else:
        filtered_kwargs = {
            "task": task,
            "tokenizer": tokenizer,
            "messages": kwargs.get("messages"),
            "prompt": kwargs.get("prompt"),
            "method": kwargs.get("method"),
            "constraints": kwargs.get("constraints"),
        }
        filtered_kwargs.update(
            {k: v for k, v in kwargs.items() if k not in filtered_kwargs}
        )
        if (
            messages
            and "constrained_beam_search" in filtered_kwargs
            and filtered_kwargs["constraints"] is None
        ):
            raise ValueError("Constrained Beam Search requires Constraints")

        response = generate_hf_response(
            checkpoint=model,
            device=device,
            **{k: v for k, v in filtered_kwargs.items() if v is not None},
        )

    return response


############################## AWS Bedrock ################################################################################################################


def create_bedrock_client(
    aws_role_name: Optional[str] = None,
    aws_session_name: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_bedrock_runtime_endpoint: Optional[str] = None,
    timeout: Optional[int] = 10,
) -> BaseClient:
    """
    Create a Bedrock Runtime client.

    Parameters:
        aws_role_name (str): The name of the AWS role to assume.
        aws_session_name (str): The name to use for the assumed role session.
        aws_access_key_id (str): The AWS access key ID to use.
        aws_secret_access_key (str): The AWS secret access key to use.
        aws_region_name (str): The AWS region to use.
        aws_bedrock_runtime_endpoint (str): The endpoint URL for the Bedrock Runtime service.
        timeout (int): Timeout value for connecting to AWS services.


    Returns:
        BaseClient: The initialized Bedrock Runtime client.

    Raises:
        ValueError: If required parameters are missing.
    """
    config = boto3.session.Config(
        connect_timeout=timeout,
        read_timeout=timeout,
    )

    if aws_role_name is not None and aws_session_name is not None:
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        sts_response = sts_client.assume_role(
            RoleArn=aws_role_name,
            RoleSessionName=aws_session_name,
        )
        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=aws_region_name,
            endpoint_url=aws_bedrock_runtime_endpoint,
            config=config,
        )
    elif aws_access_key_id is not None:
        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
            endpoint_url=aws_bedrock_runtime_endpoint,
            config=config,
        )
    else:
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region_name,
            endpoint_url=aws_bedrock_runtime_endpoint,
            config=config,
        )

    return client


def handle_service_data(
    service: str, updated_kwargs: dict, model: Optional[str] = None
):
    config = Configurations.get_config(service, model)
    payload = {}

    if service == "anthropic":
        if model.startswith("anthropic.claude-3"):
            for k, v in config.items():
                if k not in updated_kwargs:
                    updated_kwargs[k] = v
            system_prompt_idx = next(
                (
                    idx
                    for idx, message in enumerate(updated_kwargs.get("messages", []))
                    if message.get("role") == "system"
                ),
                None,
            )
            if system_prompt_idx is not None:
                updated_kwargs["system"] = updated_kwargs["messages"].pop(
                    system_prompt_idx
                )["content"]
            if model.startswith("claude-instant-1") or model.startswith("claude-2"):
                updated_kwargs["messages"] = anthropic_prompt(
                    messages=updated_kwargs.get("messages", [])
                )
            updated_kwargs["messages"] = anthropic_messages_prompt(
                messages=updated_kwargs.get("messages", [])
            )
            payload = json.dumps(
                {"messages": updated_kwargs.get("messages", []), **updated_kwargs}
            )
        else:
            for k, v in config.items():
                if k not in updated_kwargs:
                    updated_kwargs[k] = v
            payload = json.dumps(
                {"prompt": updated_kwargs.get("prompt", ""), **updated_kwargs}
            )
    else:
        if service == "titan":
            payload = json.dumps(
                {
                    "inputText": updated_kwargs.get("prompt", ""),
                    "textGenerationConfig": updated_kwargs,
                }
            )
        else:
            payload = json.dumps(
                {"prompt": updated_kwargs.get("prompt", ""), **updated_kwargs}
            )

    return payload


class Configurations:
    @staticmethod
    def get_config(service: str, model: Optional[str] = None):
        if service == "ai21":
            return AmazonBedrockAI21Configuration.config()
        elif service == "anthropic":
            if model and model.startswith("anthropic.claude-3"):
                return AmazonBedrockAnthropicClaude3Configuration.config()
            else:
                return AmazonBedrockAnthropicConfiguration.config()
        elif service == "cohere":
            return AmazonBedrockCohereConfiguration.config()
        elif service == "meta":
            return AmazonBedrockLlamaConfiguration.config()
        elif service == "mistral":
            return AmazonBedrockMistralConfiguration.config()
        elif service == "titan":
            return AmazonBedrockTitanConfiguration.config()
        else:
            return {}


####################################################### Sagemaker ########################################################################################
def create_sagemaker_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
) -> BaseClient:
    """
    Create a Sagemaker Runtime client.

    Parameters:
        aws_access_key_id (str): The AWS access key ID to use.
        aws_secret_access_key (str): The AWS secret access key to use.
        aws_region_name (str): The AWS region to use.


    Returns:
        BaseClient: The initialized Sagemaker Runtime client.

    Raises:
        ValueError: If required parameters are missing.
    """

    if aws_access_key_id is not None:
        client = boto3.client(
            service_name="sagemaker-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region_name,
        )
    else:
        client = boto3.client(
            service_name="sagemaker-runtime",
            region_name=aws_region_name,
        )

    return client


def handle_sagemaker_service_data(**updated_kwargs: dict):
    messages = updated_kwargs.get("messages")
    prompt = updated_kwargs.get("prompt")
    # Create an instance of SagemakerConfiguration
    configuration = SagemakerConfiguration()
    # Call the config method on the instance
    config = configuration.config()
    for k, v in config.items():
        if k not in updated_kwargs:
            updated_kwargs[k] = v
    if messages:
        messages = anthropic_messages_prompt(messages)
        prompt = generate_custom_prompt(
            role_dict={"roles": "user"},
            messages=messages,
        )
    if "messages" in updated_kwargs:
        updated_kwargs.pop("messages")
    elif "prompt" in updated_kwargs:
        updated_kwargs.pop("prompt")

    payload = json.dumps({"inputs": prompt, "parameters": updated_kwargs}).encode(
        "utf-8"
    )
    return payload


############################################################## Gemini, Vertex and Palm #################################################################
################################################################ Gemini ################################################################################
def load_image_from_url(url):
    """
    Load image from URL.

    Args:
        url (str): URL of the image.

    Returns:
        PIL.Image.Image: The loaded image.
    """
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def gemini_vision_convert_messages(messages):
    """
    Converts given messages for GPT-4 Vision to Gemini format.

    Args:
        messages (list): The messages to convert. Each message can be a dictionary with a "content" key. The content can be a string or a list of elements. If it is a string, it will be concatenated to the prompt. If it is a list, each element will be processed based on its type:
            - If the element is a dictionary with a "type" key equal to "text", its "text" value will be concatenated to the prompt.
            - If the element is a dictionary with a "type" key equal to "image_url", its "image_url" value will be added to the list of images.

    Returns:
        list: A list containing the prompt (a string) and the processed images (PIL.Image.Image objects).
    """
    try:
        prompt = ""
        processed_images = []

        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                for element in message["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            prompt += element["text"]
                        elif element["type"] == "image_url":
                            image_url = element["image_url"]["url"]
                            if image_url.startswith("https://"):
                                image = load_image_from_url(image_url)
                                processed_images.append(image)
                            else:
                                image = Image.open(image_url)
                                processed_images.append(image)

        content = [prompt] + processed_images
        return content
    except Exception as e:
        raise Exception(f"Gemini image conversion failed: {e}")


def gemini_text_image_prompt(messages):
    """
    Generates content in Gemini format containing text and image parts.

    Args:
        messages (list): List of messages. Each message can contain text or image URLs.

    Returns:
        list: Gemini content containing text and image parts.
    """

    prompt = ""
    images = []

    for message in messages:
        if isinstance(message["content"], str):
            prompt += message["content"]
        elif isinstance(message["content"], list):
            for element in message["content"]:
                if isinstance(element, dict):
                    if element.get("type") == "text":
                        prompt += element.get("text", "")
                    elif element.get("type") == "image_url":
                        image_url = element.get("image_url", {}).get("url", "")
                        images.append(image_url)

    # Construct content in Gemini format
    content = [{"parts": [{"text": prompt}]}, {"parts": []}]
    for image_url in images:
        if image_url.startswith("https://"):
            response = requests.get(image_url)
            if response.ok:
                data = base64.b64encode(response.content).decode("utf-8")
                image_part = {"inline_data": {"mime_type": "image/jpeg", "data": data}}
                content[1]["parts"].append(image_part)
        else:
            # Handle local file paths
            try:
                with open(image_url, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                    image_part = {
                        "inline_data": {"mime_type": "image/jpeg", "data": data}
                    }
                    content[1]["parts"].append(image_part)
            except FileNotFoundError:
                print(f"File '{image_url}' not found. Skipping.")

    return content


def handle_gemini_service_data(updated_kwargs):
    model = updated_kwargs.get("model")
    messages = updated_kwargs.get("messages")

    if model == "gemini-pro-vision":
        prompt = gemini_vision_convert_messages(messages)
    else:
        prompt = gemini_text_image_prompt(messages)

    # Merge updated_kwargs with default Gemini configuration
    config = GeminiConfiguration.config()
    updated_kwargs = {**config, **updated_kwargs}

    # Extract safety_settings from updated_kwargs and convert to genai.types.SafetySettingDict
    safety_settings_param = updated_kwargs.pop("safety_settings", None)
    safety_settings = (
        [genai.types.SafetySettingDict(x) for x in safety_settings_param]
        if safety_settings_param
        else None
    )

    return safety_settings, prompt


############################################################## Palm ######################################################################################


def update_palm_kwargs_with_config(updated_kwargs: dict):
    """
    Update `updated_kwargs` dictionary with values from `config` dictionary if the keys are not already present.

    Args:
        updated_kwargs (dict): The dictionary to be updated.
        config (dict): The dictionary containing default configuration values.

    Returns:
        None
    """
    config = PalmConfiguration.config()
    for k, v in config.items():
        if k not in updated_kwargs:
            updated_kwargs[k] = v
    return updated_kwargs


def generate_prompt_from_messages(messages: list) -> str:
    """
    Generate a prompt string from a list of messages.

    Args:
        messages (list): List of messages. Each message can contain a 'content' key.

    Returns:
        str: The generated prompt string.
    """
    prompt = ""
    for message in messages:
        if "content" in message:
            prompt += message["content"]
    return prompt


############################################################### VertexAI #################################################################################
def update_vertex_kwargs_with_config_and_safety_settings(updated_kwargs):
    config = VertexConfiguration.config()
    for k, v in config.items():
        if k not in updated_kwargs:
            updated_kwargs[k] = v

    safety_settings = updated_kwargs.pop("safety_settings", None)
    if safety_settings is not None:
        if not isinstance(safety_settings, list):
            raise ValueError("safety_settings must be a list")
        if not all(isinstance(setting, dict) for setting in safety_settings):
            raise ValueError("safety_settings must be a list of dicts")
        safety_settings = [
            gapic_content_types.SafetySetting(setting) for setting in safety_settings
        ]

    return updated_kwargs, safety_settings


def extract_prompt_and_images(messages):
    prompt = ""
    images = []

    for message in messages:
        content = message.get("content")

        if isinstance(content, str):
            prompt += content
        elif isinstance(content, list):
            for element in content:
                if isinstance(element, dict):
                    if element.get("type") == "text":
                        prompt += element.get("text", "")
                    elif element.get("type") == "image_url":
                        image_url = element.get("image_url", {}).get("url")
                        if image_url:
                            images.append(image_url)
    return prompt, images


def process_images(images):
    processed_images = []

    for img in images:
        if "gs://" in img:
            part_mime = "image/png" if "png" in img else "image/jpeg"
            google_clooud_part = Part.from_uri(img, mime_type=part_mime)
            processed_images.append(google_clooud_part)
        elif "https://" in img:
            image = load_image_from_url(img)
            processed_images.append(image)
        elif ".mp4" in img and "gs://" in img:
            part_mime = "video/mp4"
            google_clooud_part = Part.from_uri(img, mime_type=part_mime)
            processed_images.append(google_clooud_part)
        elif "base64" in img:
            _, encoded_data = img.split(",", 1)
            decoded_data = base64.b64decode(encoded_data)
            mime_type_match = re.match(r"data:(.*?);base64", img)
            mime_type = mime_type_match.group(1) if mime_type_match else "image/jpeg"
            processed_image = Part.from_data(data=decoded_data, mime_type=mime_type)
            processed_images.append(processed_image)

    return processed_images


def extract_prompt_and_images(messages):
    prompt = ""
    images = []

    for message in messages:
        content = message.get("content")

        if isinstance(content, str):
            prompt += content
        elif isinstance(content, list):
            for element in content:
                if isinstance(element, dict):
                    if element.get("type") == "text":
                        prompt += element.get("text", "")
                    elif element.get("type") == "image_url":
                        image_url = element.get("image_url", {}).get("url")
                        if image_url:
                            images.append(image_url)

    processed_images = process_images(images)
    return prompt, processed_images


########################################## Handle responses ########################################


def request_process_response(
    response, prompt=None, messages=None, provider=None, model=None, **updated_kwargs
):
    if hasattr(response, "iter_lines"):
        return process_stream_response(
            response,
            prompt=prompt if not messages else None,
            messages=messages,
            provider=provider,
            model=model,
            **updated_kwargs,
        )
    else:
        return process_non_stream_response(
            response,
            prompt=prompt if not messages else None,
            messages=messages,
            provider=provider,
            model=model,
            **updated_kwargs,
        )


def build_response(
    response_dict, data_dict, final_content, system_fingerprint, usage_info
):
    response_dict.update(
        {
            "id": data_dict.get("id", ""),
            "object": data_dict.get("object", ""),
            "created": data_dict.get("created", ""),
            "model": data_dict.get("model", ""),
            "system_fingerprint": system_fingerprint,
            "choices": [
                {
                    "index": data_dict.get("index", ""),
                    "message": {
                        "role": "assistant",
                        "content": final_content,
                        "tool_calls": data_dict.get("tools", []),
                    },
                    "finish_reason": data_dict.get("finish_reason", ""),
                    "logprobs": data_dict.get("logprobs", 0),
                }
            ],
            "usage": usage_info,
        }
    )


def process_non_stream_response(
    response, prompt=None, messages=None, provider=None, model=None, **updated_kwargs
):
    response_dict = {}
    try:
        response_dict = json.loads(response.text)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    response_model = build_response_model(
        response_dict,
        prompt=prompt if not messages else None,
        messages=messages,
        provider=provider,
        model=model,
        **updated_kwargs,
    )

    return response_model


def process_stream_response(
    response, prompt=None, messages=None, provider=None, model=None, **updated_kwargs
):
    response_dict = {}
    results = []
    usage_info = {}
    system_fingerprint = None
    final_content = ""
    if provider in ("groq", "togetherai"):
        for message in response.iter_lines():
            try:
                message = message.decode("utf-8")
                if message == "[DONE]":
                    break
                elif message.startswith("data: {"):
                    data_dict = json.loads(message[6:])
                    if provider == "groq":
                        process_data(
                            data_dict,
                            results,
                            usage_info,
                        )
                    else:
                        data_dict.update({"usage": {}})
                        togetherai_process_data(
                            data_dict,
                            results,
                            usage_info,
                            prompt=prompt if not messages else None,
                            messages=messages,
                        )

            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print("Error:", e)
        print()
    else:
        for message in response.iter_lines():
            data_dict = json.loads(message)
            if provider == "ai21":
                response_dict.update(data_dict)
                if prompt:
                    print(
                        data_dict["completions"][0]["data"]["text"], end="", flush=True
                    )
                else:
                    print(data_dict["outputs"][0]["text"], end="", flush=True)
            elif provider == "alephalpha":
                response_dict.update(data_dict)
                print(data_dict["completions"][0]["completion"], end="", flush=True)
            elif provider == "ollama":
                response_dict.update(data_dict)
                print(data_dict["message"]["content"], end="", flush=True)
            elif provider in ("openai", "azure", "perplexity"):
                response_dict.update(data_dict)
                print(data_dict["choices"][0]["delta"]["content"], end="", flush=True)

        print()

    build_response(
        response_dict,
        data_dict,
        usage_info=usage_info,
        final_content=final_content,
        system_fingerprint=system_fingerprint,
    )

    response_model = build_response_model(
        response_dict,
        prompt=prompt if not messages else None,
        messages=messages,
        provider=provider,
        model=model,
        **updated_kwargs,
    )

    return response_model


def process_non_stream_response(
    response, prompt=None, messages=None, provider=None, model=None, **updated_kwargs
):
    response_dict = {}
    try:
        response_dict = json.loads(response.text)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)

    response_model = build_response_model(
        response_dict,
        prompt if not messages else None,
        messages,
        provider,
        model,
        **updated_kwargs,
    )
    return response_model


######################################## GROQ #############################################


def process_data(data_dict, results, usage_info):
    content = data_dict.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
        results.append(content)
    usage = data_dict.get("x_groq", {}).get("usage", {})
    usage_info.update(
        {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    )


def togetherai_process_data(data_dict, results, usage_info, prompt=None, messages=None):
    encoding = tiktoken.get_encoding("cl100k_base")
    final_content = ""
    content = data_dict.get("choices", [{}])[0].get("delta", {}).get("content", "")
    if content:
        print(content, end="", flush=True)
        results.append(content)
        final_content = "".join(content)

    prompt_tokens = (
        len(encoding.encode(str([message["content"] for message in messages])))
        if messages
        else len(encoding.encode(str(prompt)))
    )
    completion_tokens = len(encoding.encode(str(final_content)))

    usage_info.update(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
    )
