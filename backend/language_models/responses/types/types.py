from typing import Dict, List, Optional, Union
from typing_extensions import Literal, Required
from openai._models import BaseModel as OpenAIBaseModel
from ..utilities import generate_prefixed_hex_id

__all__ = [
    "TopLogprob",
    "Image",
    "ChatResponseTokenLogprob",
    "ChoiceLogprobs",
    "ResponseUsage",
    "EmbeddingsUsage",
    "Function",
    "ChatResponseMessageToolCall",
    "ChatResponseMessage",
]


class TopLogprob(OpenAIBaseModel):
    """
    Represents the top log probabilities for a token.

    Attributes:
        token (str): The token for which top log probabilities are represented.
        bytes (Optional[List[int]]): Byte representation of the token. Defaults to None.
        logprob (float): The log probability of the token.
    """

    token: str
    bytes: Optional[List[int]] = None
    logprob: float


class ChatResponseTokenLogprob(OpenAIBaseModel):
    """
    Represents the log probabilities for a token in a chat response.

    Attributes:
        token (str): The token for which log probabilities are represented.
        logprob (float): The log probability of the token.
        top_logprobs (List[TopLogprob]): List of top log probabilities associated with the token.
    """

    token: str
    logprob: float
    top_logprobs: List[TopLogprob]


class ChoiceLogprobs(OpenAIBaseModel):
    """
    Represents the log probabilities for choices in a response.

    Attributes:
        content (Optional[List[ChatResponseTokenLogprob]]): List of log probabilities for response choices.
    """

    content: Optional[List[ChatResponseTokenLogprob]]


class ResponseUsage(OpenAIBaseModel):
    """
    Represents the usage of tokens in a response.

    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        completion_tokens (int): Number of tokens in the completion.
        total_tokens (int): Total number of tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsUsage(OpenAIBaseModel):
    """
    Represents the usage of embeddings in a response.

    Attributes:
        prompt_tokens (int): Number of tokens in the prompt.
        total_tokens (int): Total number of tokens.
    """

    prompt_tokens: int = 0
    total_tokens: int = 0


class Function(OpenAIBaseModel):
    """
    Represents a function call.

    Attributes:
        name (Optional[str]): Name of the function.
        arguments (Optional[str]): Arguments passed to the function.
    """

    name: Optional[str]
    arguments: Optional[str]

    def __init__(
        self,
        name: Optional[str] = None,
        arguments: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            arguments=arguments,
        )


class ChatResponseMessageToolCall(OpenAIBaseModel):
    """
    Represents a tool call within a chat response message.

    Attributes:
        id (str): Identifier for the tool call.
        index (int): Index of the tool call.
        function (Union[Dict, Function]): Function associated with the tool call.
        type (Optional[Literal["function"]]): Type of the tool call. Defaults to "function".
    """

    def __init__(
        self,
        index: int,
        id: Optional[str],
        function: Optional[Union[Dict, Function]] = None,
        type: Optional[Literal["function"]] = None,
        required: Optional[List[str]] = None,
    ):
        super().__init__(
            index=index, id=id, function=function, type=type, required=required
        )

        if isinstance(function, Dict):
            self.function = Function(**function)

        if id is None:
            self.id = generate_prefixed_hex_id("call_", 9)

        if type is None:
            self.type = "function"
        if required is None:
            required = required


class ChatResponseMessage(OpenAIBaseModel):
    """
    Represents a message in a chat response.

    Attributes:
        role (Optional[Literal["system", "user", "assistant", "tool"]]): Role of the message sender.
        content (Optional[str]): Content of the message.
        functions (Optional[Function]): Function associated with the message.
        tools (Optional[List[ChatResponseMessageToolCall]]): List of tool calls within the message.
    """

    def __init__(
        self,
        role: Optional[Literal["system", "user", "assistant", "tool"]] = None,
        content: Optional[str] = None,
        functions: Optional[Function] = None,
        tools: Optional[List[ChatResponseMessageToolCall]] = None,
        **kwargs,
    ):
        super().__init__(
            role=role,
            content=content,
            function=functions,
            tools=tools,
            **kwargs,
        )


class Image(OpenAIBaseModel):
    """
    Represents data related to an image response.

    Attributes:
        b64_json (Optional[str]): Base64 encoded JSON data representing the image.
        revised_prompt (Optional[str]): Revised prompt for generating the image response.
        url (Optional[str]): URL of the generated image.
    """

    b64_json: Optional[str]
    revised_prompt: Optional[str]
    url: Optional[str]

    def __init__(
        self,
        b64_json: Optional[str] = None,
        revised_prompt: Optional[str] = None,
        url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes an ImageResponseData object.

        Args:
            b64_json (Optional[str]): Base64 encoded JSON data representing the image.
            revised_prompt (Optional[str]): Revised prompt for generating the image response.
            url (Optional[str]): URL of the generated image.
            **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        super().__init__(
            b64_json=b64_json,
            revised_prompt=revised_prompt,
            url=url,
            **kwargs,
        )
