import os
import subprocess
import tempfile
from typing import Dict, Tuple
import docker
from ...base.base import Parameters, Tool
from ._dockercheckertool import DockerCheckerTool

DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "rust_executor"
TOOL_DESCRIPTION = (
    "This tool allows you to build and execute Rust code. "
    "Specify your Rust code in the 'code' parameter and it will compile and run the code."
    "Note that you MUST provide the 'code' parameter to use this tool."
)


class RustExecutorTool(Tool):
    """
    A tool for building and executing Rust code in a Docker container.
    """

    def __init__(self, image: str, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT):
        """
        Initialize the RustExecutorTool.

        :param image: The Docker image to use for the Rust execution environment.
        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        """
        super().__init__(
            TOOL_NAME, TOOL_DESCRIPTION, parameters=Parameters, dangerous=True
        )
        self._image = image
        self._execution_timeout = execution_timeout
        self._client = docker.from_env()
        self._docker_checker = DockerCheckerTool(
            image=image, execution_timeout=execution_timeout
        )

    def run(self, arguments: Dict) -> str:
        """
        Run the Rust code specified in the arguments.

        :param arguments: A dictionary containing the 'code' parameter.
        :return: The result of the Rust code execution.
        """
        try:
            code = arguments.get("code")

            if code is None:
                raise ValueError(
                    "To use the 'rust_executor' tool you must provide the 'code' parameter."
                )

            if self._docker_checker.check_docker_running():
                return self.execute_code_in_docker(code)
            else:
                return self.execute_code_locally(code)
        except Exception as e:
            return f"Error running RustExecutorTool: {e}"
        finally:
            self._client.close()

    def execute_code_in_docker(self, code: str) -> str:
        """
        Execute the Rust code in a Docker container.

        :param code: The Rust code to execute.
        :return: The result of the Rust code execution.
        """
        try:
            container = self._client.containers.run(
                self._image,
                stdin_open=True,
                detach=True,
                remove=True,
                command=[
                    "sh",
                    "-c",
                    f"echo '{code}' > main.rs && rustc main.rs && ./main",
                ],
                tty=True,
            )

            exit_code = container.wait(timeout=self._execution_timeout)
            if exit_code.get("StatusCode") != 0:
                raise Exception("Code execution failed")

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8")

            return stdout
        finally:
            container.remove()

    def execute_code_locally(self, code: str) -> str:
        """
        Execute the Rust code locally.

        :param code: The Rust code to execute.
        :return: The result of the Rust code execution.
        """
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".rs", delete=False
            ) as tmpfile:
                tmpfile.write(code)
                tmpfile_path = tmpfile.name

            # Compile the Rust code
            subprocess.run(["rustc", tmpfile_path], check=True)

            # Run the compiled binary
            result = subprocess.run(
                ["./" + os.path.splitext(os.path.basename(tmpfile_path))[0]],
                capture_output=True,
                text=True,
            )
            stdout = result.stdout
            stderr = result.stderr

            if result.returncode != 0:
                raise Exception("Code execution failed")
        finally:
            # Clean up the temporary file
            os.remove(tmpfile_path)

        return stdout
