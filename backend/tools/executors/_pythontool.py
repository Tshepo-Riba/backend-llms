import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import traceback

from ...base.base import Parameters, Tool

DEFAULT_EXECUTION_TIMEOUT = 60
TOOL_NAME = "python_repl"
TOOL_DESCRIPTION = (
    "This tool allows you to execute Python code. "
    "Specify your Python code in the 'script' parameter and it will return the result."
    "Note that you MUST provide the 'script' parameter to use this tool."
)


def execute_target(conn: Connection, code: str) -> None:
    """
    Execute the Python code in a subprocess.

    :param conn: The connection for communication.
    :param code: The Python code to execute.
    """
    local_vars = {}
    stdout_str, stderr_str = "", ""

    try:
        with io.StringIO() as stdout_io, io.StringIO() as stderr_io, redirect_stdout(
            stdout_io
        ), redirect_stderr(stderr_io):
            exec(code, {}, local_vars)
            stdout_str = stdout_io.getvalue()
            stderr_str = stderr_io.getvalue()

        # Remove non-picklable objects from local_vars if any
        for key in list(local_vars.keys()):
            if not isinstance(local_vars[key], (int, float, str, list, dict, tuple)):
                del local_vars[key]

        conn.send([local_vars, stdout_str, stderr_str])
    except Exception as e:
        conn.send([str(e), stdout_str, stderr_str])


class PythonREPLTool(Tool):
    """
    A tool for executing Python code in a REPL fashion.

    :param execution_timeout: Timeout for code execution in seconds (default is 60).
    """

    def __init__(self, execution_timeout: int = DEFAULT_EXECUTION_TIMEOUT) -> None:
        """
        Initialize the PythonREPLTool.

        :param execution_timeout: Timeout for code execution in seconds (default is 60).
        """
        super().__init__(
            TOOL_NAME, TOOL_DESCRIPTION, parameters=Parameters, dangerous=True
        )
        self._execution_timeout = execution_timeout

    def run(self, arguments: Dict) -> str:
        """
        Run the Python code specified in the arguments.

        :param arguments: A dictionary containing the 'script' parameter.
        :return: The result of the Python code execution.
        """
        script = arguments.get("code")

        if script is None:
            raise ValueError(
                "To use the 'python_repl' tool you must provide the 'code' parameter."
            )

        result, stdout, stderr = self.execute_code(script)

        response = ""

        # Include stdout if present
        if stdout:
            response += f"\n{stdout}\n"

        # Include stderr if present
        if stderr:
            response += f":\n{stderr}\n"

        # Include result if present
        if result:
            formatted_result = "\n".join(
                [f"{key}: {value}" for key, value in result.items()]
            )
            response += f"\n{formatted_result}\n"

        if not response:
            response = "Code executed successfully. No output or result variables."

        return response

    def execute_code(self, code: str) -> (Dict, str, str):  # type: ignore
        """
        Execute the Python code.

        :param code: The Python code to execute.
        :return: A tuple containing the result variables, stdout, and stderr outputs.
        """
        # Create a pipe for communication
        parent_conn, child_conn = Pipe()

        try:
            process = Process(target=execute_target, args=(child_conn, code))
            process.start()
            process.join(timeout=self._execution_timeout)

            if process.is_alive():
                process.terminate()
                process.join(10)
                raise TimeoutError("Code execution timed out")

            result, stdout, stderr = parent_conn.recv()
            if isinstance(result, str):
                raise Exception(result)

            return result, stdout, stderr
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            parent_conn.close()
            child_conn.close()
