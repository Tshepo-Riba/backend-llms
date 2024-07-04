from typing import Dict, Optional
from pydantic import BaseModel
from ...base.base import (
    LangchainToolProtocol,
    LangChainTools,
    InputParameters,
)


class LangChainTool(LangChainTools):
    """
    Represents a language chain tool that wraps another tool.

    :param tool: The tool to wrap.
    :param parameters: The parameters for the tool (default is InputParameters).
    :param dangerous: Flag indicating if the tool is dangerous (default is False).
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: BaseModel = None,
        dangerous: bool = False,
        tool: LangchainToolProtocol = None,
    ) -> None:
        """
        Initialize the LangChainTool.

        :param tool: The tool to wrap.
        :param parameters: The parameters for the tool (default is InputParameters).
        :param dangerous: Flag indicating if the tool is dangerous (default is False).
        """
        super().__init__(
            name=name,
            description=description,
            parameters=parameters or InputParameters(),
            dangerous=dangerous,
        )
        self._tool = tool

    def run(self, arguments: Dict) -> str:
        """
        Run the wrapped tool with the given arguments.

        :param arguments: The arguments for the tool.
        :return: The result of running the tool as a string.
        """
        try:
            return self._tool.run(arguments)
        except Exception as e:
            return f"Error running LangChainTool: {e}"
        finally:
            # Add any cleanup code here
            pass

    def description(self) -> str:
        """
        Return the description of the Tool.
        """
        return self._description

    def name(self) -> str:
        """
        Return the name of the Tool.
        """
        return self._name
