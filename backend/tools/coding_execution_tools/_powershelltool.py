import platform
import docker
from typing import Dict, Tuple
from ...base.base import Parameters, Tool

DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "powershell_executor"
TOOL_DESCRIPTION = (
    "This tool allows you to execute PowerShell scripts. "
    "Specify your PowerShell script in the 'script' parameter and it will run the script."
    "Note that you MUST provide the 'script' parameter to use this tool."
)


class PowerShellExecutorTool(Tool):
    """
    A tool for executing PowerShell scripts in a Docker container.
    """

    def __init__(self, image: str, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT):
        """
        Initialize the PowerShellExecutorTool.

        :param image: The Docker image to use for the PowerShell execution environment.
        :param execution_timeout: Timeout for script execution in seconds (default is 60).
        """
        super().__init__(
            TOOL_NAME, TOOL_DESCRIPTION, parameters=Parameters, dangerous=True
        )
        self._image = image
        self._execution_timeout = execution_timeout
        self._client = docker.from_env()
        self._check_supported_os()

    def _check_supported_os(self):
        """
        Check if the current operating system is supported.
        """
        system = platform.system()
        if system not in ["Windows", "Linux"] or (
            system == "Linux" and "microsoft" not in platform.uname().release
        ):
            raise RuntimeError(
                "PowerShellExecutorTool is only supported on Windows and Windows Subsystem for Linux (WSL)"
            )

    def run(self, arguments: Dict) -> str:
        """
        Run the PowerShell script specified in the arguments.

        :param arguments: A dictionary containing the 'script' parameter.
        :return: The result of the PowerShell script execution.
        """
        try:
            script = arguments.get("script")

            if script is None:
                raise ValueError(
                    "To use the 'powershell_executor' tool you must provide the 'script' parameter."
                )

            stdout, stderr = self.execute_script(script)

            response = ""

            # Include stdout if present
            if stdout:
                response += f"Output:\n{stdout}\n"

            # Include stderr if present
            if stderr:
                response += f"Errors:\n{stderr}\n"

            if not response:
                response = "Script executed successfully. No output or errors."

            return response
        except Exception as e:
            return f"Error running PowerShellExecutorTool: {e}"
        finally:
            self._client.close()

    def execute_script(self, script: str) -> Tuple[str, str]:
        """
        Execute the PowerShell script in a Docker container.

        :param script: The PowerShell script to execute.
        :return: A tuple containing the stdout and stderr outputs.
        """
        try:
            # Create a container
            container = self._client.containers.run(
                self._image,
                stdin_open=True,
                detach=True,
                remove=True,
                command=["powershell", "-Command", script],
                tty=True,
            )

            # Wait for the container to finish
            exit_code = container.wait(timeout=self._execution_timeout)
            if exit_code.get("StatusCode") != 0:
                raise Exception("Script execution failed")

            # Get the output
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

            return stdout, stderr
        finally:
            container.remove()
