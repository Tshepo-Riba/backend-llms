import json
import os
import subprocess
import docker
import logging
from typing import Dict, Tuple
from ._dockercheckertool import DockerCheckerTool
from ..executors._pythontool import PythonREPLTool

DEFAULT_EXECUTION_TIMEOUT = 60

logger = logging.getLogger(__name__)


class DockerPythonREPLExecutor:
    """
    A class for executing Python code in a Docker container using the PythonREPLTool.
    """

    def __init__(
        self, image: str, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT
    ) -> None:
        """
        Initialize the DockerPythonREPLExecutor.

        :param image: The Docker image to use for the Python execution environment.
        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        """
        self._image = image
        self._execution_timeout = execution_timeout
        self._client = docker.from_env()
        self._docker_checker = DockerCheckerTool()

    def execute_code(
        self, arguments: Dict, use_docker: bool = True
    ) -> Tuple[Dict, str, str]:
        """
        Execute the PythonREPLTool inside a Docker container if Docker is installed,
        otherwise execute locally.

        :param arguments: A dictionary containing the tool arguments.
        :return: A tuple containing the result variables, stdout, and stderr outputs.
        """
        # Check if the image exists locally, if not pull it
        if not self._image_exists_locally(self._image):
            self._pull_image(self._image)

        if use_docker:
            try:
                if self._docker_checker.check_docker_running():
                    # Use Docker for execution
                    print("Docker is running, executing code in Docker")
                    return self.execute_docker(arguments)
            except Exception as e:
                logger.error(f"Error checking Docker: {e}")

        # Use local execution if Docker is not available or there's an error
        print("Docker is not running or not requested, executing code locally")
        return self.execute_local(arguments)

    def _image_exists_locally(self, image: str) -> bool:
        """
        Check if the Docker image exists locally.

        :param image: The name of the Docker image.
        :return: True if the image exists locally, False otherwise.
        """
        try:
            self._client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            return False

    def _pull_image(self, image: str) -> None:
        """
        Pull the Docker image if it doesn't exist locally.

        :param image: The name of the Docker image.
        """
        print(f"Pulling Docker image: {image}")
        self._client.images.pull(image)

    def execute_docker(self, arguments: Dict) -> Tuple[Dict, str, str]:
        """
        Execute the Python code inside a Docker container.

        :param arguments: A dictionary containing the tool arguments.
        :return: A tuple containing the result variables, stdout, and stderr outputs.
        """
        # Serialize the tool arguments
        serialized_arguments = json.dumps(arguments)

        # Create a new container
        container = self._client.containers.create(
            self._image,
            stdin_open=True,
            command=[
                "python",
                "-c",
                f"exec({serialized_arguments}['code'])",
            ],
            tty=True,
        )

        try:
            # Start the container
            container.start()

            # Wait for the container to finish
            container.wait()

            # Get the output
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

            # Log the output
            logger.info("Execution complete")
            logger.debug("Standard Output: %s", stdout)
            logger.debug("Standard Error: %s", stderr)

            return {"stdout": stdout, "stderr": stderr}, stdout, stderr
        finally:
            # Remove the container
            container.remove()

    def execute_local(self, arguments: Dict) -> Tuple[Dict, str, str]:
        """
        Execute the PythonREPLTool locally.

        :param arguments: A dictionary containing the tool arguments.
        :return: A tuple containing the result variables, stdout, and stderr outputs.
        """
        process = None  # Initialize process to None
        try:
            # Set the PYTHONPATH to include the parent directory of main.py
            current_path = os.getcwd()
            parent_directory = os.path.dirname(current_path)
            os.environ["PYTHONPATH"] = parent_directory

            # Serialize the tool arguments
            serialized_arguments = arguments

            # Construct the command to execute the tool
            command = [
                "python3",
                "-c",
                f"from {PythonREPLTool.__module__} import {PythonREPLTool.__name__}; tool_instance = {PythonREPLTool.__name__}(); print(tool_instance.run({serialized_arguments}))",
            ]

            # Execute the command
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            exit_code = process.returncode

            if exit_code != 0:
                raise Exception(
                    f"Code execution failed with exit code {exit_code}: {stderr.decode('utf-8')}"
                )

            # Log the output
            logger.info("Execution complete")
            logger.debug("Standard Output: %s", stdout.decode("utf-8"))
            logger.debug("Standard Error: %s", stderr.decode("utf-8"))

            return (
                {"stdout": stdout.decode("utf-8"), "stderr": stderr.decode("utf-8")},
                stdout.decode("utf-8"),
                stderr.decode("utf-8"),
            )
        except Exception as e:
            logger.error(f"Error executing code locally: {e}")
            return {}, "", ""
        finally:
            # Ensure the process is cleaned up if it's not None
            if process:
                process.terminate()
                process.wait()
