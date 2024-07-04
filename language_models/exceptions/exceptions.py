import httpx
from typing import Optional
from urllib3.exceptions import HTTPError
from ...base.base import APIException


class GenericAPIException(APIException):
    """
    Generic exception class for API errors.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        api_url: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ) -> None:
        """
        Initialize GenericAPIException.

        Parameters:
            status_code (int): The HTTP status code of the response.
            message (str): The error message.
            api_url (str): The base URL of the API.
            request (Optional[httpx.Request]): The HTTP request.
            response (Optional[httpx.Response]): The HTTP response.
        """
        super().__init__(status_code, message, api_url)
        self.request = request
        self.response = response


class OpenAIException(GenericAPIException):
    """
    Exception class for errors from OpenAI API.
    """

    DEFAULT_URL: str = "https://api.openai.com/v1"


class AnthropicException(GenericAPIException):
    """
    Exception class for errors from Anthropic API.
    """

    DEFAULT_URL: str = "https://api.anthropic.com/v1/"


class HuggingfaceException(GenericAPIException):
    """
    Exception class for errors from Huggingface API.
    """

    DEFAULT_URL: str = "https://api-inference.huggingface.co/"


class GeminiPalmException(GenericAPIException):
    """
    Exception class for errors from Google's Gemini Palm API.
    """

    DEFAULT_URL: str = (
        "https://developers.generativeai.google/api/python/google/generativeai/"
    )


class OllamaException(GenericAPIException):
    """
    Exception class for errors from Ollama API.
    """

    DEFAULT_URL: str = "http://localhost:11434"


class VLLMException(GenericAPIException):
    """
    Exception class for errors from VLLM API.
    """

    DEFAULT_URL: str = "http://0.0.0.0:8000"


class AI21Exception(GenericAPIException):
    """
    Exception class for errors from AI21 API.
    """

    DEFAULT_URL: str = "https://api.ai21.com/studio/v1/"


class AlephAlphaException(GenericAPIException):
    """
    Exception class for errors from Aleph Alpha API.
    """

    DEFAULT_URL: str = "https://api.aleph-alpha.com"


class BedrockException(GenericAPIException):
    """
    Exception class for errors from AWS Bedrock API.
    """

    DEFAULT_URL: str = "https://us-west-2.console.aws.amazon.com/bedrock"


class SagemakerException(GenericAPIException):
    """
    Exception class for errors from AWS Sagemaker API.
    """

    DEFAULT_URL: str = "https://us-west-2.console.aws.amazon.com/sagemaker"


class CohereException(GenericAPIException):
    """
    Exception class for errors from Cohere API.
    """

    DEFAULT_URL: str = "https://api.cohere.ai/v1/"


class VertexException(GenericAPIException):
    """
    Exception class for errors from Cohere API.
    """

    DEFAULT_URL: str = "https://cloud.google.com/vertex-ai/"


class TogetherException(GenericAPIException):
    """
    Exception class for errors from Together AI API.
    """

    DEFAULT_URL: str = "https://api.together.xyz/inference"


class GroqException(GenericAPIException):
    """
    Exception class for errors from Together AI API.
    """

    DEFAULT_URL: str = "https://api.groq.com/openai/v1"


class PerplexityException(GenericAPIException):
    """
    Exception class for errors from Together AI API.
    """

    DEFAULT_URL: str = "https://api.perplexity.ai"


class ExceptionFactory:
    exceptions = {
        "openai": OpenAIException,
        "anthropic": AnthropicException,
        "huggingface": HuggingfaceException,
        "geminipalm": GeminiPalmException,
        "ollama": OllamaException,
        "vllm": VLLMException,
        "ai21": AI21Exception,
        "alephalpha": AlephAlphaException,
        "bedrock": BedrockException,
        "sagemaker": SagemakerException,
        "cohere": CohereException,
        "together": TogetherException,
        "vertex": VertexException,
        "groq": GroqException,
        "perplexity": PerplexityException,
    }

    @classmethod
    def get_exception(cls, provider):
        return cls.exceptions.get(provider.lower(), GenericAPIException)
