from enum import Enum


class ModelType(Enum):
    """
    An enumeration of different types of models.
    """

    LLM = "LLM"
    EMBEDDING = "EMBEDDING"


class ModelFamily(Enum):
    """
    An enumeration of different families of models.
    """

    GPT35 = "GPT_35"
    GPT4 = "GPT_4"
    OPEN = "OPEN"
    ANTROPIC = "ANTROPIC"
    AI21 = "AI21"
    ALEPHALPHA = "ALEPHALPHA"
    BEDROCK = "BEDROCK"
    COHERE = "COHERE"
    GOOGLE = "GOOGLE"
    GROQ = "GROQ"
    OLLAMA = "OLLAMA"
    HUGGINGFACE = "HUGGINGFACE"
    PERPLEXITY = "PERPLEXITY"
    SAGEMAKER = "SAGEMAKER"
    TOGETHERAI = "TOGETHERAI"


class ModelName(Enum):
    """
    An enumeration of different model names.
    """

    GPT4 = "gpt-4"
    GPT4_0613 = "gpt-4-0613"
    GPT4_32K = "gpt-4-32k"
    GPT4_TURBO_1106 = "gpt-4-1106-preview"
    GPT35_TURBO = "gpt-3.5-turbo"
    GPT35_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT35_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT35_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT35_TURBO_16K_0613 = "gpt-3.5-turbo-16k-0613"
    CLAUDE_2 = "claude-2"
    CLAUDE_21 = "claude-2.1"
    CLAUDE_INSTANT = "claude-instant"
    J2_LIGHT = "j2-light"
    J2_MID = "j2-mid"
    J2_ULTRA = "j2-ultra"
    LUMINOUS_BASE = "luminous-base"
    LUMINOUS_EXTENDED = "luminous-extended"
    LUMINOUS_SUPREME = "luminous-supreme"
    LUMINOUS_BASE_CONTROL = "luminous-base-control"
    LUMINOUS_EXTENDED_CONTROL = "luminous-extended-control"
    LUMINOUS_SUPREME_CONTROL = "luminous-supreme-control"
    LLAMA3 = "llama3-70b-8192"


class EmbedModelName(Enum):
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    AI21_EMBEDDING = "ai21_Embeddings_Model"
    LUMINOUS_BASE_EMBEDDING = "luminous-base"
    LUMINOUS_EXTENDED_EMBEDDING = "luminous-extended"
    LUMINOUS_SUPREME_EMBEDDING = "luminous-supreme"
    LUMINOUS_BASE_CONTROL_EMBEDDING = "luminous-base-control"
    LUMINOUS_EXTENDED_CONTROL_EMBEDDING = "luminous-extended-control"
    LUMINOUS_SUPREME_CONTROL_EMBEDDING = "luminous-supreme-control"


class ModelConfig:
    """
    A class representing a model configuration.

    Args:
    - model_type (ModelType): The type of the model.
    - family (ModelFamily): The family of the model.
    - max_tokens (int): The maximum number of tokens.
    - input_price_per_token (float): The input price per token.
    - output_price_per_token (float): The output price per token.
    - price_per_token (float): The price per token.
    - embedding_size (int): The embedding size.

    Methods:
    - create_llm_model_config(cls, family, max_tokens, input_price_per_token=None, output_price_per_token=None):
      Creates a model configuration for a language model.
    - create_embedding_model_config(cls, max_tokens, price_per_token, embedding_size):
      Creates a model configuration for an embedding model.
    """

    def __init__(
        self,
        model_type,
        family,
        max_tokens,
        input_price_per_token=None,
        output_price_per_token=None,
        price_per_token=None,
        embedding_size=None,
    ):
        self.model_type = model_type
        self.family = family
        self.max_tokens = max_tokens
        self.input_price_per_token = input_price_per_token
        self.output_price_per_token = output_price_per_token
        self.price_per_token = price_per_token
        self.embedding_size = embedding_size

    @classmethod
    def create_llm_model_config(
        cls, family, max_tokens, input_price_per_token=None, output_price_per_token=None
    ):
        """
        Create a model configuration for a language model.

        Args:
        - family (ModelFamily): The family of the model.
        - max_tokens (int): The maximum number of tokens.
        - input_price_per_token (float): The input price per token.
        - output_price_per_token (float): The output price per token.

        Returns:
        - ModelConfig: The model configuration.
        """
        return cls(
            ModelType.LLM,
            family,
            max_tokens,
            input_price_per_token,
            output_price_per_token,
        )

    @classmethod
    def create_embedding_model_config(cls, max_tokens, price_per_token, embedding_size):
        """
        Create a model configuration for an embedding model.

        Args:
        - max_tokens (int): The maximum number of tokens.
        - price_per_token (float): The price per token.
        - embedding_size (int): The embedding size.

        Returns:
        - ModelConfig: The model configuration.
        """
        return cls(
            ModelType.EMBEDDING,
            None,
            max_tokens,
            None,
            None,
            price_per_token,
            embedding_size,
        )


# Define configurations
configurations = {
    ModelName.GPT4.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT4, 8191, 0.00003, 0.00006
    ),
    ModelName.GPT4_0613.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT4, 8191, 0.00003, 0.00006
    ),
    ModelName.GPT4_32K.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT4, 32767, 0.00006, 0.00012
    ),
    ModelName.GPT4_TURBO_1106.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT4, 128000, 0.00001, 0.00003
    ),
    ModelName.GPT35_TURBO.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT35, 4095, 0.0000015, 0.000002
    ),
    ModelName.GPT35_TURBO_16K.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT35, 16383, 0.000003, 0.000004
    ),
    ModelName.GPT35_TURBO_0613.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT35, 4095, 0.0000015, 0.000002
    ),
    ModelName.GPT35_TURBO_0125.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT35, 2048, 0.0000015, 0.000002
    ),
    ModelName.GPT35_TURBO_16K_0613.value: ModelConfig.create_llm_model_config(
        ModelFamily.GPT35, 16383, 0.000003, 0.000004
    ),
    ModelName.CLAUDE_2.value: ModelConfig.create_llm_model_config(
        ModelFamily.ANTROPIC, 200000, 0.00001102, 0.00003268
    ),
    # TODO: other providers including anthropic and various
    ModelName.J2_LIGHT.value: ModelConfig.create_llm_model_config(
        ModelFamily.AI21, 100000, 0.0001, 0.0005
    ),
    ModelName.J2_MID.value: ModelConfig.create_llm_model_config(
        ModelFamily.AI21, 100000, 0.00025, 0.00125
    ),
    ModelName.J2_ULTRA.value: ModelConfig.create_llm_model_config(
        ModelFamily.AI21, 100000, 0.002, 0.010
    ),
    # TODO: fix the pricing for the below providers
    ModelName.LUMINOUS_BASE.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.03, 0.033
    ),
    ModelName.LUMINOUS_EXTENDED.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.045, 0.0495
    ),
    ModelName.LUMINOUS_SUPREME.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.175, 0.1925
    ),
    ModelName.LUMINOUS_BASE_CONTROL.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.0375, 0.04125
    ),
    ModelName.LUMINOUS_EXTENDED_CONTROL.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.05625, 0.061875
    ),
    ModelName.LUMINOUS_SUPREME_CONTROL.value: ModelConfig.create_llm_model_config(
        ModelFamily.ALEPHALPHA, 100000, 0.21875, 0.240625
    ),
    ModelName.LLAMA3.value: ModelConfig.create_llm_model_config(
        ModelFamily.GROQ, 8192, 0.00021875, 0.000240625
    ),
}


embed_configurations = {
    EmbedModelName.TEXT_EMBEDDING_ADA_002.value: ModelConfig.create_embedding_model_config(
        8190, 0.0000001, 1536
    ),
    EmbedModelName.AI21_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        8190, 0.0001, 1536
    ),
    EmbedModelName.LUMINOUS_BASE_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        100000, 0.03, 0.033
    ),
    EmbedModelName.LUMINOUS_EXTENDED_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        8190, 0.0000001, 1536
    ),
    EmbedModelName.LUMINOUS_SUPREME_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        8190, 0.0000001, 1536
    ),
    EmbedModelName.LUMINOUS_BASE_CONTROL_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        100000, 0.0375, 0.04125
    ),
    EmbedModelName.LUMINOUS_EXTENDED_CONTROL_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        8190, 0.0000001, 1536
    ),
    EmbedModelName.LUMINOUS_SUPREME_CONTROL_EMBEDDING.value: ModelConfig.create_embedding_model_config(
        8190, 0.0000001, 1536
    ),
}
