import json
import time
from typing_extensions import Literal
from typing import List, Optional
from openai._models import BaseModel as OpenAIBaseModel

from ..types.types import EmbeddingsUsage
from ..utilities import generate_prefixed_hex_id

"""
The provided information outlines the process of creating a model response for a embeddings using the OpenAI compatible objects. 
It details the necessary parameters required in the POST request to the API endpoint, including messages, model, and various optional parameters . 
The API returns a embeddings completion object.

Reference:
OpenAI. (n.d.). Chat API Documentation. Retrieved from :https://platform.openai.com/docs/api-reference/embeddings
        
"""


class EmbeddingsResponse(OpenAIBaseModel):
    """
    Represents an embeddings response from the model.

    Attributes:
        id (str): Identifier for the embeddings response.
        model (str): Name of the model generating the response.
        object (Literal["embeddings"]): Type of the response object.
        choices (List[Choice]): List of choices in the response.
        created (int): Timestamp indicating when the response was created.
        system_fingerprint (Optional[str]): System fingerprint associated with the response.
        usage (Optional[EmbeddingsUsage]): Usage statistics of the response.
    """

    id: Optional[str]
    model: Optional[str]
    object: Optional[Literal["list"]]
    data: Optional[List]
    created: Optional[int]
    usage: Optional[EmbeddingsUsage]

    def __init__(
        self,
        id: Optional[str] = None,
        model: Optional[str] = None,
        object: Optional[Literal["list"]] = "list",
        data: Optional[List] = None,
        created: Optional[int] = None,
        usage: Optional[EmbeddingsUsage] = None,
        **kwargs,
    ):
        """
        Initializes an EmbeddingsResponse object.

        Args:
            id (str): Identifier for the embeddings response.
            model (str): Name of the model generating the response.
            object (Literal["embeddings"]): Type of the response object.
            choices (List[Choice]): List of choices in the response.
            created (Optional[int]): Timestamp indicating when the response was created.
            system_fingerprint (Optional[str]): System fingerprint associated with the response.
            usage (Optional[EmbeddingsUsage]): Usage statistics of the response.
            **kwargs: Additional keyword arguments to be passed to the base class constructor.
        """
        data = [] if data is None else data

        id = generate_prefixed_hex_id("embedding-", 12)
        if created is None:
            created = int(time.time())
        if usage is None:
            usage = EmbeddingsUsage()

        super().__init__(
            id=id,
            model=model,
            object=object,
            data=data,
            created=created,
            usage=usage,
            **kwargs,
        )

    def json_serializer(self):
        """
        Serialize the delta object to JSON.

        Returns:
            str: JSON representation of the delta object.
        """
        return json.dumps(self.dict(), default=lambda o: o.__dict__)

    def __str__(self):
        """
        String representation of the delta object.

        Returns:
            str: String representation of the delta object.
        """
        return self.json_serializer()

    def __contains__(self, item):
        """
        Check if an attribute exists in the delta object.

        Args:
            item: Attribute name.

        Returns:
            bool: True if the attribute exists, False otherwise.
        """
        return hasattr(self, item)

    def __getitem__(self, key):
        """
        Get the value of an attribute in the delta object.

        Args:
            key: Attribute name.

        Returns:
            Any: Value of the attribute.
        """
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """
        Set the value of an attribute in the delta object.

        Args:
            key: Attribute name.
            value: Value to be set.
        """
        setattr(self, key, value)

    def __eq__(self, other):
        """
        Check if two delta objects are equal.

        Args:
            other: Another delta object.

        Returns:
            bool: True if the two objects are equal, False otherwise.
        """
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        return False

    def __delattr__(self, item):
        """
        Delete an attribute from the delta object.

        Args:
            item: Attribute name to be deleted.
        """
        delattr(self, item)

    def __len__(self):
        """
        Get the number of attributes in the delta object.

        Returns:
            int: Number of attributes.
        """
        return len(self.__dict__)

    def copy(self, other):
        """
        Copy attributes from another delta object.

        Args:
            other: Another delta object.
        """
        for attr, value in other.__dict__.items():
            setattr(self, attr, value)

    def add(self, name, value):
        """
        Add an attribute to the delta object.

        Args:
            name: Name of the attribute.
            value: Value to be assigned to the attribute.
        """
        setattr(self, name, value)
