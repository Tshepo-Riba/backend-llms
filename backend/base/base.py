from abc import ABC, abstractmethod
from pydantic import BaseModel, Field  # type: ignore
from typing import Any, Dict, List, Optional, Protocol, Union, TypeVar

import httpx  # type: ignore

T = TypeVar("T")


class InputParameters(BaseModel):
    """
    A class representing input parameters.

    Attributes:
        input (str): The input string.
        context (str): The context string.
    """

    input: str = ""
    context: str = ""


class EmptyParameters(BaseModel):
    """
    Represents an empty set of parameters.
    """


class Tool:
    """
    Represents a tool with a name, description, parameters, and optional flags.

    :param name: The name of the tool.
    :param description: The description of the tool.
    :param parameters: The parameters of the tool.
    :param dangerous: Flag indicating if the tool is dangerous (default is False).
    :param supports_streaming: Flag indicating if the tool supports streaming (default is False).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[BaseModel] = None,
        dangerous: bool = False,
        supports_streaming: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize a Tool instance.

        :raises ValueError: If name or description is empty.
        """
        if not name:
            raise ValueError("name is required")
        if not description:
            raise ValueError("description is required")

        self.name = name
        self.description = description
        self.parameters = parameters or EmptyParameters()
        self.dangerous = dangerous
        self.supports_streaming = supports_streaming

    def get_tool_function(self) -> Dict:
        """
        Get the tool function information.

        :return: A dictionary containing the tool function information.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters.schema(),
        }

    def run(self, arguments: Dict) -> Union[str, Dict]:
        """
        Run the tool with the given arguments.

        :param arguments: The arguments for the tool.
        :return: The result of running the tool.
        """
        return arguments.get("input", "")

    @staticmethod
    def handle_stream_update(arguments_buffer: str) -> None:
        """
        Handle a stream update.

        This function is called when the agent is streaming data to the tool.
        The arguments_buffer is a string buffer containing the latest argument data that has been received.
        """
        pass

    @staticmethod
    def save_state() -> Dict:
        """
        Save the state of the tool.

        :return: The state of the tool as a dictionary.
        """
        return {}

    @staticmethod
    def load_state(state: Dict) -> None:
        """
        Load the state of the tool.

        :param state: The state of the tool as a dictionary.
        """
        pass

    @staticmethod
    def stop() -> None:
        """
        Stop the tool.

        This function is called when the agent is being forcibly stopped.
        """
        pass


class Parameters(BaseModel):
    """
    Parameters for the PythonREPLTool.

    :param script: The Python script to execute.
    :param context: Optional parameter for additional context or contexts.
    """

    script: str
    context: str = ""


class LangchainToolProtocol(Protocol):
    """
    A protocol representing a language chain tool.

    Methods:
    - name(self) -> str: Get the name of the tool.
    - description(self) -> str: Get the description of the tool.
    - run(self, arguments: Dict) -> str: Run the tool with the given arguments.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    def run(self, arguments: Dict) -> str:
        pass


class LangChainTools(LangchainToolProtocol):
    """
    Represents a tool with a name, description, and parameters.
    """

    name: str
    description: str
    parameters: BaseModel

    def run(self, arguments: Dict) -> str:
        """
        Run the tool with the given arguments.

        :param arguments: The arguments for the tool.
        :return: The result of running the tool as a string.
        """
        pass


class TemplateRenderer:
    """A base class for rendering templates."""

    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render a template with given context.

        :param template: The template string.
        :param context: The context to render the template.
        :return: The rendered template.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ClientResponse(ABC):
    """A base class for representing client responses."""

    class Choices(ABC):
        """A nested abstract base class for representing choices."""

        class Messages(ABC):
            """A nested abstract base class for representing messages."""

            @property
            @abstractmethod
            def content(self) -> Optional[str]:
                """Return the content of the message.

                :return: The content of the message.
                """
                pass

        @property
        @abstractmethod
        def message(self) -> Messages:
            """Return the message.

            :return: The message.
            """
            pass

    @property
    @abstractmethod
    def choices(self) -> List[Choices]:
        """Return the choices.

        :return: The list of choices.
        """
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the model.

        :return: The model.
        """
        pass


class LLM(ABC):
    """A base class for defining client interfaces."""

    def create(self, params: Dict[str, Any]) -> ClientResponse:
        """Create a resource with given parameters.

        :param params: The parameters for creating the resource.
        :return: The response from the client.
        """
        pass

    def embed(self, params: Dict[str, Any]) -> Any:
        """Create a resource with given parameters.

        :param params: The parameters for creating the resource.
        :return: The response from the client.
        """
        pass

    def messages(
        self, response: ClientResponse
    ) -> Union[List[str], List[ClientResponse.Choices.Messages]]:
        """Retrieve messages from the response.

        :param response: The response from the client.
        :return: The list of messages.
        """
        pass

    def cost(self, response: ClientResponse) -> float:
        """Calculate the cost based on the response.

        :param response: The response from the client.
        :return: The cost calculated based on the response.
        """
        pass

    def usage(self, response: ClientResponse) -> Dict:
        """Retrieve usage information from the response.

        :param response: The response from the client.
        :return: The usage information.
        """
        pass


class Cache(Protocol):
    """A protocol for defining cache interfaces."""

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get value from cache.

        :param key: The key to retrieve the value.
        :param default: Default value if key not found. Defaults to None.
        :return: The value associated with the key.
        """
        pass

    def set(self, key: str, value: str) -> None:
        """Set value in cache.

        :param key: The key to set the value.
        :param value: The value to be set.
        """
        pass

    def close(self) -> None:
        """Close the cache."""
        pass

    def __enter__(self) -> "Cache":
        """Enter the context."""
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context."""
        pass


class APIException(Exception):
    """
    Base class for API exceptions.
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        url: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
    ) -> None:
        """
        Initialize APIException.

        :param status_code: The HTTP status code of the response.
        :param message: The error message.
        :param url: The URL that caused the exception.
        :param request: The HTTP request.
        :param response: The HTTP response.
        """
        self.status_code = status_code
        self.message = message
        self.url = url
        self.request = request
        self.response = response
        super().__init__(f"{status_code} - {message}")

        # Initialize default request and response if not provided
        self._init_default_response()

    def _init_default_request(self) -> None:
        """
        Initialize default HTTP request.
        """
        if self.request is None:
            self.request = httpx.Request(method="POST", url=self.url)

    def _init_default_response(self) -> None:
        """
        Initialize default HTTP response.
        """
        if self.response is None:
            self._init_default_request()
            self.response = httpx.Response(
                status_code=self.status_code, request=self.request
            )

    def __str__(self) -> str:
        """
        Return a string representation of the exception.
        """
        return f"{type(self).__name__}: {self.status_code} - {self.message}"


class ConfigurationBase:
    """
    Base class for configuration objects.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes a configuration object with provided keyword arguments.

        Args:
            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        # If kwargs are provided, update instance variables
        for key, value in kwargs.items():
            setattr(self, key, value)

    def config(self) -> Dict:
        """
        Returns a dictionary representation of the configuration.

        Returns:
            Dict: Dictionary containing non-None instance variables.
        """
        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("__") and v is not None
        }
