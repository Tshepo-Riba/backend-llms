import logging
import random
import string
import uuid
from typing import List


embedding_models: List = ["text-embedding-ada-002"]


def print_verbose(message: str):
    """
    Prints a verbose message using the logger.

    Args:
        message (str): The message to print.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.debug(message)


def generate_prefixed_hex_id(prefix: str, length: int) -> str:
    """
    Generate a hexadecimal ID with a given prefix and length.

    Args:
        prefix (str): The prefix to be included in the hexadecimal ID.
        length (int): The length of the random hexadecimal part of the ID.

    Returns:
        str: The generated hexadecimal ID with the specified prefix.
    """
    hex_characters = string.hexdigits[
        :-6
    ]  # Exclude lowercase letters to ensure consistency
    random_hex = "".join(random.choice(hex_characters) for _ in range(length))
    return prefix + random_hex


def get_finish_reason(finish_reason: str) -> str:
    """
    Map finish reasons to standardized values.

    Args:
        finish_reason (str): The original finish reason to be mapped.

    Returns:
        str: The standardized finish reason.
    """
    # Define mapping of finish reasons to standardized values
    mapping = {
        "stop_sequence": "stop",
        "ERROR": "stop",
        "end_turn": "stop",
        "eos_token": "stop",
        "max_tokens": "length",
        "length": "length",
        "function_call": "function_call",
        "content_filter": "content_filter",
        "null": "null",
        "tool_calls": "tool_calls",
    }

    # Return the mapped value or the original value if not found in mapping
    return mapping.get(finish_reason, finish_reason)


def filter_openai_completion_params(params: dict) -> dict:
    """
    Filters the input dictionary to include only keys present in _openai_completion_params.

    Args:
        params (dict): The input dictionary containing parameters.

    Returns:
        dict: A dictionary containing only keys present in _openai_completion_params.
    """
    mapping = [
        "functions",
        "function_call",
        "temperature",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "request_timeout",
        "api_base",
        "api_version",
        "api_key",
        "deployment_id",
        "organization",
        "base_url",
        "default_headers",
        "timeout",
        "response_format",
        "seed",
        "tools",
        "tool_choice",
        "max_retries",
    ]

    return {key: value for key, value in params.items() if key in mapping}


def filter_completion_params(params: dict) -> dict:
    """
    Filters the input dictionary to include only keys present in _litellm_completion_params.

    Args:
        params (dict): The input dictionary containing parameters.

    Returns:
        dict: A dictionary containing only keys present in _litellm_completion_params.
    """
    mapping = [
        "metadata",
        "acompletion",
        "caching",
        "mock_response",
        "api_key",
        "api_version",
        "api_base",
        "force_timeout",
        "logger_fn",
        "verbose",
        "provider",
        "litellm_logging_obj",
        "litellm_call_id",
        "use_client",
        "id",
        "fallbacks",
        "azure",
        "headers",
        "model_list",
        "num_retries",
        "context_window_fallback_dict",
        "roles",
        "final_prompt_value",
        "bos_token",
        "eos_token",
        "request_timeout",
        "complete_response",
        "self",
        "client",
        "rpm",
        "tpm",
        "input_cost_per_token",
        "output_cost_per_token",
        "hf_model_name",
        "model_info",
        "proxy_server_request",
        "preset_cache_key",
    ]

    return {key: value for key, value in params.items() if key in mapping}
