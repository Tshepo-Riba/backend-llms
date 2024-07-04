import docker
from typing import Dict
from ...base.base  import Parameters, Tool

DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "docker_checker"
TOOL_DESCRIPTION = (
    "This tool checks if Docker is installed and, if not, executes locally."
)


class DockerCheckerTool(Tool):
    """
    A tool for checking if Docker is installed and running.
    """

    def __init__(self):
        """
        Initialize the DockerCheckerTool.
        """
        super().__init__(
            TOOL_NAME, TOOL_DESCRIPTION, parameters=Parameters, dangerous=False
        )
        self._client = docker.from_env()

    def run(self, arguments: Dict) -> str:
        """
        Run the Docker checker and return the status.

        :param arguments: A dictionary containing the 'code' parameter.
        :return: The result of the execution.
        """
        try:
            # Check if Docker is running
            docker_running = self.check_docker_running()
            return "Docker is running" if docker_running else "Docker is not running"
        except Exception as e:
            return f"Error running Docker checker: {e}"
        finally:
            self._client.close()

    def check_docker_running(self) -> bool:
        """
        Check if Docker is running.

        :return: True if Docker is running, False otherwise.
        """
        try:
            self._client.ping()
            return True
        except docker.errors.APIError:
            return False