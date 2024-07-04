import requests
import re
import xml.etree.ElementTree as Element
import json
from typing import List, Optional
from enum import Enum
from jinja2 import Environment, exceptions
from PIL import Image
from io import BytesIO


def default_prompt(messages):
    return " ".join(message["content"] for message in messages)


def generate_custom_prompt(
    role_dict: dict,
    messages: list,
    initial_prompt: str = "",
    final_prompt: str = "",
    bos_token: str = "",
    eos_token: str = "",
):
    prompt_parts = [bos_token + initial_prompt] if initial_prompt else [bos_token]
    bos_open = True

    for message in messages:
        role = message["role"]
        role_info = role_dict.get(role, {})  # Retrieve role info once

        if role in {"system", "human"} and not bos_open:
            prompt_parts.append(bos_token)
            bos_open = True

        prompt_parts.append(message["content"])

        if role == "assistant":
            prompt_parts.append(eos_token)
            bos_open = False

    prompt_parts.append(final_prompt)
    return "".join(prompt_parts)


def extract_required_keys(parameters):
    required_keys = set()
    if "required" in parameters:
        required_keys.update(parameters["required"])
    if "properties" in parameters:
        for key, value in parameters["properties"].items():
            if isinstance(value, dict):
                required_keys.update(extract_required_keys(value))
    return required_keys


def format_example(parameters):
    example = "{\n"
    for key, value in parameters.items():
        example += f'  "{key}": "{value}",\n'
    example += "}"
    return example


def format_properties(properties):
    formatted_props = []
    for key, value in properties.items():
        if "description" in value:
            formatted_props.append(f"{key}: {value['description']}")
        else:
            formatted_props.append(key)
    return ", ".join(formatted_props)


def function_call_prompt(messages: list, tools: list):
    function_prompt = (
        ".Produce JSON OUTPUT ONLY! The following functions are available to you:\n"
    )
    for tool in tools:
        if tool.get("type") == "function":
            function_name = tool["function"]["name"]
            description = tool["function"]["description"]
            parameters = tool["function"]["parameters"]

            # Extract required keys recursively
            required_keys = extract_required_keys(parameters)
            required_params = ", ".join(required_keys)

            # Format example JSON
            example_json = format_example({key: "" for key in required_keys})

            # Format properties
            properties = tool["function"]["parameters"].get("properties", {})
            formatted_properties = format_properties(properties)

            function_prompt += f"- {function_name}: {description} (Required parameters: {required_params}, Properties: {formatted_properties}) Example: {example_json}. \n Return only the {required_params}. \n Output only the {required_params}. \n"

    # Include additional queries
    additional_queries = [
        message["content"]
        for message in messages
        if "role" in message and message["role"] == "query"
    ]
    if additional_queries:
        function_prompt += "\nAdditional Queries:\n"
        for query in additional_queries:
            function_prompt += f"- {query}\n"

    function_added_to_prompt = any(
        "user" in message.get("role", "") for message in messages
    )
    if not function_added_to_prompt:
        messages.append({"role": "user", "content": function_prompt})
    else:
        for message in messages:
            if "user" in message.get("role", ""):
                message["content"] += function_prompt

    return messages


def ai21_message_prompt(messages):
    converted_messages = []
    system_message = ""
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
        else:
            new_message = {}
            new_message["text"] = message["content"]
            new_message["role"] = message["role"]
            converted_messages.append(new_message)
    return converted_messages, system_message


def alephalpha_message_prompt(messages):
    """
    Aggregate messages from the user role into a single sentence with punctuation marks.

    Args:
        messages (list): A list of dictionaries containing messages.

    Returns:
        str: A single sentence containing aggregated messages from the user role with punctuation marks.
    """
    user_messages = [
        message["content"] for message in messages if message["role"] == "user"
    ]
    return " ".join(user_messages) + "."


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


def anthropic_prompt(prompt):
    human_prompt = AnthropicConstants.HUMAN_PROMPT.value
    ai_prompt = AnthropicConstants.AI_PROMPT.value
    modified_prompt = f"{human_prompt}{prompt}, Claude{ai_prompt}"
    return modified_prompt


def anthropic_messages_prompt(messages: list):
    """
    You can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
    """

    prompt_parts = []

    for idx, message in enumerate(messages):
        if message["role"] == "user":
            prompt_parts.append(
                f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
            )
        elif message["role"] == "system":
            prompt_parts.append(
                f"{AnthropicConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
            )
        else:
            prompt_parts.append(
                f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
            )

        if idx == 0 and message["role"] == "assistant":
            # Ensure the prompt always starts with `\n\nHuman: `
            prompt_parts[0] = (
                f"{AnthropicConstants.HUMAN_PROMPT.value}{prompt_parts[0][len(AnthropicConstants.AI_PROMPT.value):]}"
            )

    if messages[-1]["role"] != "assistant":
        prompt_parts.append(f"{AnthropicConstants.AI_PROMPT.value}")

    return "".join(prompt_parts)


def claude_2_1_prompt(messages: list):
    """
    Reference , you can "put words in Claude's mouth" by ending with an assistant message.
    See: https://docs.anthropic.com/claude/docs/put-words-in-claudes-mouth
    """
    prompt_parts = []

    for idx, message in enumerate(messages):
        if message["role"] == "user":
            prompt_parts.append(
                f"{AnthropicConstants.HUMAN_PROMPT.value}{message['content']}"
            )
        elif message["role"] == "system":
            if idx > 0 and messages[idx - 1]["role"] != "system":
                prompt_parts.append(AnthropicConstants.HUMAN_PROMPT.value)
            prompt_parts.append(message["content"])
        elif message["role"] == "assistant":
            if idx > 0 and messages[idx - 1]["role"] == "system":
                prompt_parts.append(AnthropicConstants.HUMAN_PROMPT.value)
            prompt_parts.append(
                f"{AnthropicConstants.AI_PROMPT.value}{message['content']}"
            )

    if messages[-1]["role"] != "assistant":
        prompt_parts.append(AnthropicConstants.AI_PROMPT.value)

    return "".join(prompt_parts)


def cohere_message_prompt(messages):
    user_content = ""
    for message in messages:
        if message.get("role") == "user":
            user_content += message.get("content") + " "
    return user_content.strip()


def prompt_to_messages(prompt, system_content="You are a helpful AI assistant."):
    messages = []
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt})

    return messages


def message_to_prompt(messages):
    prompt = ""
    for message in messages:
        if "role" in message and message["role"] == "user":
            prompt += message["content"]
    return prompt


class AmazonTitanConstants(Enum):
    HUMAN_PROMPT = "\n\nUser: "
    AI_PROMPT = "\n\nBot: "


def amazon_titan_prompt(messages: list):
    prompt = ""
    for idx, message in enumerate(messages):
        if message["role"] == "system":
            prompt += f"{AmazonTitanConstants.HUMAN_PROMPT.value}<admin>{message['content']}</admin>"
        else:
            prompt += f"{AmazonTitanConstants.HUMAN_PROMPT.value if message['role'] == 'user' else AmazonTitanConstants.AI_PROMPT.value}{message['content']}"
        if idx == 0 and message["role"] == "assistant":
            prompt = AmazonTitanConstants.HUMAN_PROMPT.value + prompt
    if messages[-1]["role"] != "assistant":
        prompt += AmazonTitanConstants.AI_PROMPT.value
    return prompt


def mistral_instruct_prompt(messages):
    role_info_dict = {
        "system": (["[INST] \n", " [/INST]\n"]),
        "user": (["[INST] ", " [/INST]\n"]),
        "assistant": ([" ", "</s> "]),
    }
    return generate_custom_prompt(
        role_info_dict, messages, initial_prompt="<s>", final_prompt=""
    )


def convert_messages_to_prompt(model, messages, service, custom_prompt_dict):
    if service in ("anthropic", "amazon"):
        model_prompt_details = custom_prompt_dict[model]
        prompt = generate_custom_prompt(
            role_dict=model_prompt_details["roles"],
            initial_prompt_value=model_prompt_details["initial_prompt_value"],
            final_prompt_value=model_prompt_details["final_prompt_value"],
            messages=messages,
        )
    else:
        if "amazon.titan-text" in model:
            prompt = amazon_titan_prompt(messages=messages)
        elif "anthropic." in model:
            if any(_ in model for _ in ["claude-2.1", "claude-v2:1"]):
                prompt = claude_2_1_prompt(messages=messages)
            else:
                return anthropic_prompt(messages=messages)
        elif "mistral." in model:
            prompt = mistral_instruct_prompt(messages=messages)

    return prompt


def llama_2_chat_prompt(messages):
    role_info_dict = {
        "system": {"pre": "[INST] <<SYS>>\n", "post": "\n<</SYS>>\n [/INST]\n"},
        "user": {"pre": "[INST] ", "post": " [/INST]\n"},
        "assistant": {"pre": "\n"},
    }
    return generate_custom_prompt(
        role_info_dict, messages, bos_token="<s>", eos_token="</s>"
    )


def ollama_prompt(model, messages):
    if "instruct" in model:
        role_info_dict = {
            "system": {"pre": "### System:\n", "post": "\n"},
            "user": {"pre": "### User:\n", "post": "\n"},
            "assistant": {"pre": "### Response:\n", "post": "\n"},
        }
        return generate_custom_prompt(
            role_info_dict, messages, final_prompt="### Response:"
        )
    elif "llava" in model:
        prompt = ""
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                for element in message["content"]:
                    if isinstance(element, dict) and element["type"] == "text":
                        prompt += element["text"]
                    elif isinstance(element, dict) and element["type"] == "image_url":
                        images.append(element["image_url"]["url"])
        return {"prompt": prompt, "images": images}
    else:
        return "".join(
            m["content"] if isinstance(m["content"], str) else "".join(m["content"])
            for m in messages
        )


def mistral_api_prompt(messages):
    new_messages = []
    for m in messages:
        texts = ""
        if isinstance(m["content"], list):
            for c in m["content"]:
                if c["type"] == "image_url":
                    return messages
                elif c["type"] == "text" and isinstance(c["text"], str):
                    texts += c["text"]
        new_m = {"role": m["role"], "content": texts}
        new_messages.append(new_m)
    return new_messages


def falcon_instruct_prompt(messages):
    prompt = ""
    for message in messages:
        if message["role"] == "system":
            prompt += message["content"]
        else:
            prompt += "{}: {}\n\n".format(
                message["role"],
                message["content"].replace("\r\n", "\n").replace("\n\n", "\n"),
            )

    return prompt


def falcon_chat_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"{message['role'].capitalize()}: {message['content']}\n"
    return prompt


def wizardcoder_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"### {message['role'].capitalize()}:\n{message['content']}\n\n"
    return prompt


def phind_codellama_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"### {message['role'].capitalize()}\n{message['content']}\n\n"
    return prompt


def construct_format_parameters_prompt(parameters: dict) -> str:
    return "".join(f"<{key}>{value}</{key}>\n" for key, value in parameters.items())


def construct_format_tool_for_claude_prompt(
    name: str, description: str, parameters: dict
) -> str:
    parameters_str = construct_format_parameters_prompt(parameters)
    return (
        f"<tool_description>\n"
        f"<tool_name>{name}</tool_name>\n"
        f"<description>\n{description}\n</description>\n"
        f"<parameters>\n{parameters_str}</parameters>\n"
        "</tool_description>"
    )


def construct_tool_use_system_prompt(tools):
    tool_descriptions = [
        construct_format_tool_for_claude_prompt(
            tool["function"]["name"],
            tool["function"].get("description", ""),
            tool["function"].get("parameters", {}),
        )
        for tool in tools
    ]
    return (
        "In this environment you have access to a set of tools you can use to answer the user's question.\n\n"
        "You may call them like this:\n"
        "<function_calls>\n"
        "<invoke>\n"
        "<tool_name>$TOOL_NAME</tool_name>\n"
        "<parameters>\n"
        "<$PARAMElementER_NAME>$PARAMElementER_VALUE</$PARAMElementER_NAME>\n"
        "...\n"
        "</parameters>\n"
        "</invoke>\n"
        "</function_calls>\n\n"
        "Here are the tools available:\n"
        "<tools>\n" + "\n".join(tool_descriptions) + "\n</tools>"
    )


def convert_url_to_base64(url):
    import base64

    try:
        response = requests.get(url)
        response.raise_for_status()
        image_bytes = response.content
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        img_type = url.split(".")[-1].lower()
        img_types = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }

        if img_type in img_types:
            img_type = img_types[img_type]
            return f"data:{img_type};base64,{base64_image}"
        else:
            raise Exception(
                f"Error: Unsupported image format. Format={img_type}. Supported types = {list(img_types.keys())}"
            )
    except requests.RequestException as e:
        raise Exception(f"Error: Unable to fetch image from URL. {e}")


def convert_to_anthropic_image_obj(openai_image_url: str):
    if openai_image_url.startswith("http"):
        openai_image_url = convert_url_to_base64(url=openai_image_url)

    base64_data = openai_image_url.split(";base64,")[1]
    image_format = openai_image_url.split(";base64,")[0].split("/")[-1]

    return {
        "source": {
            "type": "base64",
            "media_type": f"image/{image_format}",
            "data": base64_data,
        }
    }


def anthropic_messages_prompt(messages: list):
    new_messages = []
    for message in messages:
        role = message["role"]
        if role == "user":  # Only keep messages with role "user"
            content = message["content"]
            if isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image_url":
                        new_content.append(
                            {
                                "type": "image",
                                "source": convert_to_anthropic_image_obj(
                                    item["image_url"]["url"]
                                ),
                            }
                        )
                    elif item.get("type") == "text":
                        new_content.append({"type": "text", "content": item["text"]})
                content = new_content

            new_messages.append({"role": role, "content": content})

    return new_messages


def extract_between_tags(tag: str, string: str, strip: bool = False) -> List[str]:
    extracted_list = re.findall(f"<{tag}>(.+?)</{tag}>", string, re.DOTALL)
    if strip:
        extracted_list = [e.strip() for e in extracted_list]
    return extracted_list


def parse_xml_params(xml_content: str) -> dict:
    params = {}
    root = Element.fromstring(xml_content)
    for child in root.findall(".//parameters/*"):
        params[child.tag] = child.text
    return params


def load_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        if not content_type or "image" not in content_type:
            raise ValueError(
                f"URL does not point to a valid image (content-type: {content_type})"
            )

        return Image.open(BytesIO(response.content))

    except requests.RequestException as e:
        raise Exception(f"Request failed: {e}")
    except Exception as e:
        raise e


def gemini_vision_convert_messages(messages: list):
    try:
        prompt = ""
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                for element in message["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            prompt += element["text"]
                        elif element["type"] == "image_url":
                            image_url = element["image_url"]["url"]
                            images.append(image_url)
        # processing images passed to gemini
        processed_images = []
        for img in images:
            if "https:/" in img:
                image = load_image_from_url(img)
                processed_images.append(image)
            else:
                image = Image.open(img)
                processed_images.append(image)
        return prompt, processed_images
    except Exception as e:
        raise e


def gemini_text_image_prompt(messages: list):
    try:
        prompt = ""
        images = []
        for message in messages:
            if isinstance(message["content"], str):
                prompt += message["content"]
            elif isinstance(message["content"], list):
                for element in message["content"]:
                    if isinstance(element, dict):
                        if element["type"] == "text":
                            prompt += element["text"]
                        elif element["type"] == "image_url":
                            image_url = element["image_url"]["url"]
                            images.append(image_url)

        content = [prompt] + images
        return content
    except Exception as e:
        raise e


def hf_chat_template(
    model: str,
    messages: List[dict],
    chat_template: Optional[str] = None,
    hf_api_key: Optional[str] = None,
):
    try:
        if chat_template is None:

            tokenizer_config_url = (
                f"https://huggingface.co/{model}/raw/main/tokenizer_config.json"
            )
            headers = {
                "Authorization": f"Bearer {hf_api_key}"
            }  # Replace YOUR_ACCESS_TOKEN with your Hugging Face access token if necessary
            response = requests.get(tokenizer_config_url, headers=headers)
            response.raise_for_status()
            tokenizer_config = json.loads(response.content)
            chat_template = tokenizer_config.get("chat_template")
            if not chat_template:
                raise ValueError(
                    "No chat template found in the tokenizer configuration"
                )

        env = Environment()
        env.globals["raise_exception"] = lambda message: Exception(
            f"Error message - {message}"
        )

        try:
            template = env.from_string(chat_template)
        except exceptions.TemplateSyntaxError as e:
            raise Exception(f"Error parsing template: {e}")

        def render_template(messages):
            reformatted_messages = [
                {"role": "user", "content": ""}
            ]  # Start with a user message
            for message in messages:
                if message["role"] == "system":
                    reformatted_messages.append(
                        {"role": "user", "content": message["content"]}
                    )
                else:
                    reformatted_messages.append(message)
            return reformatted_messages

        try:
            rendered_text = template.render(
                bos_token=tokenizer_config.get("bos_token", ""),
                eos_token=tokenizer_config.get("eos_token", ""),
                messages=render_template(messages),
            )
            return rendered_text.strip()
        except exceptions.TemplateError as e:
            raise Exception(f"Error rendering template: {e}")
        except ValueError as e:
            # Handle the case where no chat template is found
            return "No chat template available. Please provide a custom chat template."

    except requests.RequestException as e:
        raise Exception(f"Error fetching tokenizer config: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Error decoding JSON response: {e}")
    except ValueError as e:
        raise Exception(f"ValueError: {e}")


def prompt_factory(
    model: str,
    messages: list,
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
):
    # Convert the model name to lowercase
    model = model.lower()
    # Map provider to corresponding prompt generation function
    prompt_generators = {
        "ollama": ollama_prompt,
        "anthropic": (
            anthropic_prompt
            if model in {"claude-instant-1", "claude-2"}
            else anthropic_messages_prompt
        ),
        "gemini": (
            gemini_vision_convert_messages
            if model == "gemini-pro-vision"
            else gemini_text_image_prompt
        ),
        "mistral": mistral_api_prompt,
        "perplexity": lambda messages: [m.pop("name", None) for m in messages],
        "meta-llama/llama-2": llama_2_chat_prompt if "chat" in model else None,
        "tiiuae/falcon": (
            falcon_chat_prompt
            if model == "tiiuae/falcon-180B-chat"
            else falcon_instruct_prompt if "instruct" in model else None
        ),
        "codellama/codellama": llama_2_chat_prompt if "instruct" in model else None,
        "wizardlm/wizardcoder": wizardcoder_prompt,
        "phind/phind-codellama": phind_codellama_prompt,
        "togethercomputer/llama-2": (
            llama_2_chat_prompt if "instruct" in model or "chat" in model else None
        ),
    }

    # Check if a provider-specific prompt generation function is available
    if provider and provider in prompt_generators:
        prompt_generator = prompt_generators[provider]
        return prompt_generator(messages)

    try:
        if model.strip() == "meta-llama/llamaguard-7b":
            # Define chat template for Llamaguard model
            chat_template = "{% if messages|length % 2 == 0 %}{% set role = 'Agent' %}{% else %}{% set role = 'User' %}{% endif %}{{ \"[INST]  '\" + role + \"' messages in conversations according our safety policy with the below categories.\n\n<BEGIN UNSAFE CONTENT CATEGORIES>\nO1: Violence and Hate. \nShould not\n- Help people plan or engage in violence.\n- Encourage people to engage in violence.\n- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.\n- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.\nCan\n- Provide information on violence and discrimination.\n- Discuss the topics of hate and violence, and can discuss historical events involving violence.\nO2: Sexual Content.\nShould not\n- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.\nCan\n- Discuss topics related to sexuality and sexual education.\n- Provide general information regarding sexual activities.\n- Provide guidance on topics related to sexual health.\nO3: Criminal Planning.\nShould not\n- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in criminal activities or condone criminal behavior.\nCan\n- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.\n- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.\nO4: Guns and Illegal Weapons.\nShould not\n- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.\n- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.\nCan\n- Discuss firearms and the arguments for and against firearm ownership.\nO5: Regulated or Controlled Substances.\nShould not\n- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.\n- Assist or encourage people to create such substances when it is illegal to do so.\nCan\n- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).\n- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.\nO6: Self-Harm.\nShould not\n- Encourage people to harm themselves.\n- Romanticize or condone self-harm or suicide.\n- Provide information regarding the methods of suicide or self-harm.\n- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.\nCan\n- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).\nShould\n- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.\n<END UNSAFE CONTENT CATEGORIES>\n\n<BEGIN CONVERSATION>\n\n\" }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{% set role = 'User' %}{% elif message['role'] == 'assistant' %}{% set role = 'Agent' %}{% endif %}{{ role + ': ' + content.strip() + '\n\n' }}{% endfor %}{{ \"<END CONVERSATION>\n\nProvide your safety assessment for \" + role + \" in the above conversation:\n- First line must read 'safe' or 'unsafe'.\n- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]\" }}"
            return hf_chat_template(
                model=model, messages=messages, chat_template=chat_template
            )
        else:
            return hf_chat_template(model, messages)
    except Exception as e:
        # Return default prompt in case of exception
        return default_prompt(messages=messages)
