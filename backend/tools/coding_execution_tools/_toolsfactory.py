from ._dotnetexecutortool import DotnetExecutorTool
from ._shelltool import ShellTool
from ._rustexecutortool import RustExecutorTool
from ._powershelltool import PowerShellExecutorTool
from ._langchaintools import LangChainTool
from ._dockerpythonREPLexecutor import DockerPythonREPLExecutor
from ...base.base import LangchainToolProtocol


class ToolsFactory:
    @staticmethod
    def python_repl_executor(
        image: str, execution_timeout: int = 60
    ) -> DockerPythonREPLExecutor:
        """
        Create a DockerPythonREPLExecutor instance.

        :param image: The Docker image to use for the Python execution environment.
        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        :return: An instance of DockerPythonREPLExecutor.
        """
        try:
            return DockerPythonREPLExecutor(
                image=image, execution_timeout=execution_timeout
            )
        except Exception as e:
            return f"Error creating DockerPythonREPLExecutor: {e}"
        finally:
            pass  # No specific cleanup needed here

    @staticmethod
    def dotnet_executor(image: str, execution_timeout: int = 60) -> DotnetExecutorTool:
        """
        Create a DotnetExecutorTool instance.

        :param image: The Docker image to use for the .NET execution environment.
        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        :return: An instance of DotnetExecutorTool.
        """
        try:
            return DotnetExecutorTool(image=image, execution_timeout=execution_timeout)
        except Exception as e:
            return f"Error creating DotnetExecutorTool: {e}"
        finally:
            pass  # No specific cleanup needed here

    @staticmethod
    def langchain_tool(tool: LangchainToolProtocol) -> LangChainTool:
        """
        Create a LangChainTool instance that wraps the given tool.

        :param tool: The tool to wrap.
        :return: An instance of LangChainTool wrapping the given tool.
        """
        try:
            return LangChainTool(tool=tool)
        except Exception as e:
            return f"Error creating LangChainTool: {e}"
        finally:
            pass  # No specific cleanup needed here

    @staticmethod
    def powershell_executor(
        image: str, execution_timeout: int = 60
    ) -> PowerShellExecutorTool:
        """
        Create a PowerShellExecutorTool instance.

        :param image: The Docker image to use for the PowerShell execution environment.
        :param execution_timeout: Timeout for script execution in seconds (default is 60).
        :return: An instance of PowerShellExecutorTool.
        """
        try:
            return PowerShellExecutorTool(
                image=image, execution_timeout=execution_timeout
            )
        except Exception as e:
            return f"Error creating PowerShellExecutorTool: {e}"
        finally:
            pass  # No specific cleanup needed here

    @staticmethod
    def rust_executor(image: str, execution_timeout: int = 60) -> RustExecutorTool:
        """
        Create a RustExecutorTool instance.

        :param image: The Docker image to use for the Rust execution environment.
        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        :return: An instance of RustExecutorTool.
        """
        try:
            return RustExecutorTool(image=image, execution_timeout=execution_timeout)
        except Exception as e:
            return f"Error creating RustExecutorTool: {e}"
        finally:
            pass  # No specific cleanup needed here

    @staticmethod
    def shell_tool(executor: callable, execution_timeout: int = 60) -> ShellTool:
        """
        Create a ShellTool instance.

        :param executor: A callable object that executes shell commands.
        :param execution_timeout: Timeout for command execution in seconds (default is 60).
        :return: An instance of ShellTool.
        """
        try:
            return ShellTool(
                command_executor=executor, execution_timeout=execution_timeout
            )
        except Exception as e:
            return f"Error creating ShellTool: {e}"
        finally:
            pass  # No specific cleanup needed here
