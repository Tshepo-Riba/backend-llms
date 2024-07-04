import platform
import shlex
import subprocess
from typing import Dict

from ...base.base import Tool


DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "shell"
TOOL_DESCRIPTION = "This tool enables you to run shell commands. Provide your command in the 'command' parameter, and it will execute and return the result. Please note that this tool only accepts a single string argument ('command') and does not support a list of commands."


class ShellTool(Tool):
    """
    A tool for executing shell commands.

    :param command_executor: A callable object that executes shell commands.
    :param execution_timeout: Timeout for command execution in seconds (default is 60).
    """

    def __init__(
        self,
        command_executor: callable,
        execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT,
    ) -> None:
        """
        Initialize the ShellTool.

        :param command_executor: A callable object that executes shell commands.
        :param execution_timeout: Timeout for command execution in seconds (default is 60).
        """
        self._command_executor = command_executor
        self._execution_timeout = execution_timeout

    def run(self, arguments: Dict) -> str:
        """
        Run the shell command specified in the arguments.

        :param arguments: A dictionary containing the 'command' parameter.
        :return: The result of the shell command execution.
        """
        try:
            cmd = arguments.get("command")
            if cmd is None:
                raise ValueError("'command' parameter is required")

            stdout, stderr = self._command_executor(cmd, self._execution_timeout)

            response = ""

            # Include stdout if present
            if stdout:
                response += f"Output:\n{stdout}\n"

            # Include stderr if present
            if stderr:
                response += f"Errors:\n{stderr}\n"

            if not response:
                response = "Command executed successfully. No output."

            return response
        except Exception as e:
            return f"Error running ShellTool: {e}"
        finally:
            pass  # No specific cleanup needed here


class ShellCommandExecutor:
    """
    A class for executing shell commands.
    """

    @staticmethod
    def execute_command(cmd: str, timeout: int) -> (str, str):  # type: ignore
        """
        Execute the shell command.

        :param cmd: The shell command to execute.
        :param timeout: Timeout for command execution in seconds.
        :return: A tuple containing the stdout and stderr outputs as strings.
        """
        try:
            if platform.system() == "Windows":
                # Check if WSL is installed
                try:
                    subprocess.run(
                        ["wsl", "--help"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                    )
                except subprocess.CalledProcessError:
                    raise Exception("WSL is required but not installed.")

            process = subprocess.Popen(
                shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=timeout)
            return stdout.decode("utf-8"), stderr.decode("utf-8")
        except Exception as e:
            return "", f"Error executing command: {e}"


# # Create an instance of ShellTool using ShellCommandExecutor as the command executor
# shell_tool = ShellTool(ShellCommandExecutor.execute_command)

# # Define a command to execute
# command = "ls -l"

# # Run the command using the shell tool
# result = shell_tool.run({"command": command})

# # Print the result
# print(result)
