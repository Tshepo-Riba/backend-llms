import os
import traceback
from typing import Dict
from pydantic import BaseModel
from langchain_community.utilities import (
    ArxivAPIWrapper,
    BingSearchAPIWrapper,
    GoogleSearchAPIWrapper,
    SearchApiAPIWrapper,
    StackExchangeAPIWrapper,
)
from langchain_community.tools import BraveSearch, DuckDuckGoSearchRun, Tool
from ...base.base import InputParameters, LangchainToolProtocol
from ..coding_execution_tools._langchaintools import LangChainTool


os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""


class ArxivAPIWrapperTool(LangChainTool):
    """
    Represents a tool that uses the ArXiv API Wrapper.

    :param query: The query to search for.
    """

    def __init__(
        self,
        name: str,
        query: str,
        parameters: BaseModel = None,
        dangerous: bool = False,
        tool: LangchainToolProtocol = None,
    ) -> None:
        """
        Initialize the ArxivAPIWrapperTool.

        :param query: The query to search for.
        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        """
        super().__init__(
            name=name,
            description="Tool using the ArXiv API Wrapper.",
            parameters=parameters or InputParameters(),
            dangerous=dangerous,
            tool=tool,
        )
        self._query = query

    def run(self, arguments: Dict) -> str:
        """
        Run the ArxivAPIWrapperTool with the given arguments.

        :param arguments: The arguments for the tool (not used in this implementation).
        :return: The result of running the tool as a string.
        """
        try:
            arxiv = ArxivAPIWrapper()
            result = arxiv.run(self._query)
            return str(result)
        except Exception as e:
            return f"Error executing ArXiv API Wrapper: {str(e)}"
        finally:

            pass


class WebSearchTool(LangChainTool):
    """
    Represents a language chain tool that wraps the Bing Search API wrapper.
    """

    def __init__(
        self,
        name: str = "Web Search",
        description: str = "Web search using different engines.",
        parameters: BaseModel = None,
        dangerous: bool = False,
        tool: LangchainToolProtocol = None,
    ) -> None:
        """
        Initialize the WebSearchTool.

        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        """
        super().__init__(
            name=name,
            description=description,
            tool=DuckDuckGoSearchRun(),
            parameters=parameters or InputParameters(),
            dangerous=dangerous,
        )

    def run(self, arguments: Dict) -> str:
        """
        Run the wrapped tool with the given arguments.

        :param arguments: The arguments for the tool.
        :return: The result of running the tool as a string.
        """
        try:
            engine = arguments.get(
                "engine", "duckduckgo"
            )  # Set DuckDuckGo as the default engine
            if engine == "brave":
                brave_search = BraveSearch.from_api_key(
                    api_key="API KEY", search_kwargs={"count": 3}
                )
                return brave_search.run(arguments.get("query"))
            elif engine == "duckduckgo":
                duckduckgo_search = DuckDuckGoSearchRun()
                return duckduckgo_search.run(arguments.get("query"))
            elif engine == "google":
                google_search = Tool(
                    name="google_search",
                    description="Search Google for recent results.",
                    func=GoogleSearchAPIWrapper().run,
                )
                return google_search.run(arguments.get("query"))
            elif engine == "searchapi":
                searchapi_search = SearchApiAPIWrapper()
                return searchapi_search.run(arguments.get("query"))
            elif engine == "bing":
                bing_search = BingSearchAPIWrapper()
                return bing_search.run(arguments.get("query"))
            else:  # Default to DuckDuckGo
                return super().run(arguments)
        except Exception as e:
            return f"Error executing WebSearchTool: {str(e)}"


class StackExchangeTool(LangChainTool):
    """
    Represents a tool that interacts with the Stack Exchange API.

    :param parameters: The parameters for the tool (default is InputParameters).
    :param dangerous: Flag indicating if the tool is dangerous (default is False).
    """

    def __init__(
        self,
        name: str = "Stack Exchange Search",
        description: str = "Stack Exhange search.",
        parameters: BaseModel = None,
        dangerous: bool = False,
        tool: LangchainToolProtocol = None,
    ) -> None:
        """
        Initialize the StackExchangeTool.

        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        """
        super().__init__(
            name=name,
            description=description,
            tool=StackExchangeAPIWrapper(),
            parameters=parameters or InputParameters(),
            dangerous=dangerous,
        )

    def run(self, arguments: Dict) -> str:
        """
        Run the Stack Exchange tool with the given arguments.

        :param arguments: The arguments for the tool.
        :return: The result of running the tool as a string.
        """
        try:
            return self._tool.run(arguments)
        except Exception as e:
            return f"Error executing StackExchangeTool: {str(e)}"


class LangToolsFactory:
    @staticmethod
    def arxiv_api_wrapper_tool(
        query: str, parameters: BaseModel = None, dangerous: bool = False
    ) -> ArxivAPIWrapperTool:
        """
        Create an ArxivAPIWrapperTool.

        :param query: The query to search for.
        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        :return: An instance of ArxivAPIWrapperTool.
        """
        try:
            return ArxivAPIWrapperTool(
                query=query,
                parameters=parameters or InputParameters(),
                dangerous=dangerous,
            )
        except Exception as e:
            print(f"Error creating ArxivAPIWrapperTool: {str(e)}")

    @staticmethod
    def web_search_tool(
        parameters: BaseModel = None, dangerous: bool = False
    ) -> WebSearchTool:
        """
        Create a WebSearchTool.

        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        :return: An instance of WebSearchTool.
        """
        try:
            return WebSearchTool(
                parameters=parameters or InputParameters(), dangerous=dangerous
            )
        except Exception as e:
            print(f"Error creating WebSearchTool: {str(e)}")

    @staticmethod
    def stack_exchange_tool(
        parameters: BaseModel = None, dangerous: bool = False
    ) -> StackExchangeTool:
        """
        Create a StackExchangeTool.

        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        :return: An instance of StackExchangeTool.
        """
        try:
            return StackExchangeTool(
                parameters=parameters or InputParameters(), dangerous=dangerous
            )
        except Exception as e:
            print(f"Error creating StackExchangeTool: {str(e)}")
