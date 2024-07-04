import subprocess
from typing import Dict, Tuple
import docker
from ...base.base import Parameters, Tool
from ._dockercheckertool import DockerCheckerTool

DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "dotnet_executor"
TOOL_DESCRIPTION = (
    "This tool allows you to build and execute .NET code. "
    "Specify your .NET code in the 'code' parameter and it will compile and run the code."
    "Note that you MUST provide the 'code' parameter to use this tool."
)


class DotnetExecutorTool(Tool):
    """
    A tool for building and executing .NET code in a Docker container.
    """

    def __init__(self, image: str, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT):
        """
        Initialize the DotnetExecutorTool.

        :param image: The Docker image to use for the .NET execution environment.
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
        Run the .NET code specified in the arguments.

        :param arguments: A dictionary containing the 'code' parameter.
        :return: The result of the .NET code execution.
        """
        try:
            code = arguments.get("code")

            if code is None:
                raise ValueError(
                    "To use the 'dotnet_executor' tool you must provide the 'code' parameter."
                )

            if self._docker_checker.check_docker_running():
                return self.execute_code_in_docker(code)
            else:
                return self.execute_code_locally(code)
        except Exception as e:
            return f"Error running dotnet_executor tool: {e}"
        finally:
            self._client.close()

    def execute_code_in_docker(self, code: str) -> str:
        """
        Execute the .NET code in a Docker container.

        :param code: The .NET code to execute.
        :return: The result of the .NET code execution.
        """
        try:
            container = self._client.containers.run(
                self._image,
                stdin_open=True,
                detach=True,
                remove=True,
                command=["sh", "-c", f"echo '{code}' > Program.cs && dotnet run"],
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
        Execute the .NET code locally.

        :param code: The .NET code to execute.
        :return: The result of the .NET code execution.
        """
        try:
            with open("Program.cs", "w") as file:
                file.write(code)

            # Execute the code using dotnet run
            process = subprocess.Popen(
                ["dotnet", "run"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                raise Exception(f"Code execution failed: {stderr.decode('utf-8')}")

            return stdout.decode("utf-8")
        finally:
            subprocess.run(["rm", "Program.cs"])


# # Initialize the DotnetExecutorTool with a Docker image
# dotnet_tool = DotnetExecutorTool(image="your_dotnet_image")

# # Define the code to execute
# code_to_execute = """
# using System;

# public class Program
# {
#     public static void Main()
#     {
#         Console.WriteLine("Hello, world!");
#     }
# }
# """

# # Run the tool with the code
# result = dotnet_tool.run({"code": code_to_execute})

# # Print the result
# print(result)