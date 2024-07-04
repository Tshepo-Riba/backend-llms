import re


class Validations:
    """
    A class for performing validations.
    """

    @staticmethod
    def validate_api_keys(api_keys: dict) -> dict:
        """
        Validate API keys based on predefined patterns.

        :param api_keys: A dictionary containing API keys.
        :return: A dictionary mapping keys to validation results or error messages.
        """
        validations = {}
        for key, value in api_keys.items():
            try:
                key_re = None
                if key.lower() == "openai":
                    key_re = re.compile(r"^sk-[A-Za-z0-9]{32,}$")
                elif key.lower() == "google":
                    key_re = re.compile(r"^AIza[0-9A-Za-z-_]{35}$")
                elif key.lower() in ["serpapi", "azure", "aws"]:
                    key_re = re.compile(r"^[0-9A-Za-z-_]{32,}$")
                elif key.lower() == "huggingface":
                    key_re = re.compile(r"^hf_[0-9A-Za-z-_]{32,}$")
                else:
                    raise ValueError(f"Unsupported API key type: {key}")
                validations[key] = bool(re.fullmatch(key_re, value))
            except ValueError as e:
                validations[key] = f"Error: {str(e)}"
        return validations
