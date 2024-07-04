import types
from typing import Optional, Union, List, Dict
from ...base.base import ConfigurationBase


class OpenAIConfiguration(ConfigurationBase):
    """
    Configuration class for OpenAI's text generation API.

    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(
        self,
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[int] = None,
        functions: Optional[List[str]] = None,
        logit_bias: Optional[dict] = None,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        suffix: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        tools: Optional[List[Dict[str, Union[str, int]]]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for OpenAI API.

        Args:
            best_of (Optional[int]): Controls sampling. Higher values mean the model will consider more options before picking the most likely output.
            echo (Optional[bool]): If true, the response will include the prompt as part of the response.
            frequency_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            functions (Optional[List[str]]): List of function names from which the generated code will be called.
            logit_bias (Optional[dict]): Dict mapping tokens (strings) to multiplier (float) values to bias the generation probability.
            logprobs (Optional[int]): Controls how much the model should attend to the current input when generating the output.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            n (Optional[int]): Controls the number of samples to generate.
            presence_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            stop (Optional[Union[str, List[str]]]): One or more tokens where generation is stopped.
            suffix (Optional[str]): The suffix to append to the prompt.
            temperature (Optional[int]): Controls the randomness of the output.
            top_p (Optional[int]): Controls the cumulative probability of the model's output.
            tools (Optional[List[Dict[str, Union[str, int]]]]): List of tool name, version pairs.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.best_of = best_of
        self.echo = echo
        self.frequency_penalty = frequency_penalty
        self.functions = functions
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.suffix = suffix
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools


class PerplexityConfiguration(ConfigurationBase):

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for OpenAI API.

        Args:
            best_of (Optional[int]): Controls sampling. Higher values mean the model will consider more options before picking the most likely output.
            echo (Optional[bool]): If true, the response will include the prompt as part of the response.
            frequency_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            functions (Optional[List[str]]): List of function names from which the generated code will be called.
            logit_bias (Optional[dict]): Dict mapping tokens (strings) to multiplier (float) values to bias the generation probability.
            logprobs (Optional[int]): Controls how much the model should attend to the current input when generating the output.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            n (Optional[int]): Controls the number of samples to generate.
            presence_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            stop (Optional[Union[str, List[str]]]): One or more tokens where generation is stopped.
            suffix (Optional[str]): The suffix to append to the prompt.
            temperature (Optional[int]): Controls the randomness of the output.
            top_p (Optional[int]): Controls the cumulative probability of the model's output.
            tools (Optional[List[Dict[str, Union[str, int]]]]): List of tool name, version pairs.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.frequency_penalty = frequency_penalty
        self.max_tokens = max_tokens
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class GroqConfiguration(ConfigurationBase):
    """
    Configuration class for OpenAI's text generation API.

    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for OpenAI API.

        Args:
            best_of (Optional[int]): Controls sampling. Higher values mean the model will consider more options before picking the most likely output.
            echo (Optional[bool]): If true, the response will include the prompt as part of the response.
            frequency_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            functions (Optional[List[str]]): List of function names from which the generated code will be called.
            logit_bias (Optional[dict]): Dict mapping tokens (strings) to multiplier (float) values to bias the generation probability.
            logprobs (Optional[int]): Controls how much the model should attend to the current input when generating the output.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            n (Optional[int]): Controls the number of samples to generate.
            presence_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            stop (Optional[Union[str, List[str]]]): One or more tokens where generation is stopped.
            suffix (Optional[str]): The suffix to append to the prompt.
            temperature (Optional[int]): Controls the randomness of the output.
            top_p (Optional[int]): Controls the cumulative probability of the model's output.
            tools (Optional[List[Dict[str, Union[str, int]]]]): List of tool name, version pairs.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p


class AzureOpenAIConfiguration(ConfigurationBase):
    """
    Configuration class for OpenAI's text generation API.

    Reference: https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        functions: Optional[List[str]] = None,
        logit_bias: Optional[dict] = None,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        tools: Optional[List[Dict[str, Union[str, int]]]] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for OpenAI API.

        Args:
            best_of (Optional[int]): Controls sampling. Higher values mean the model will consider more options before picking the most likely output.
            echo (Optional[bool]): If true, the response will include the prompt as part of the response.
            frequency_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            functions (Optional[List[str]]): List of function names from which the generated code will be called.
            logit_bias (Optional[dict]): Dict mapping tokens (strings) to multiplier (float) values to bias the generation probability.
            logprobs (Optional[int]): Controls how much the model should attend to the current input when generating the output.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            n (Optional[int]): Controls the number of samples to generate.
            presence_penalty (Optional[int]): Helps control the diversity of tokens in the response.
            stop (Optional[Union[str, List[str]]]): One or more tokens where generation is stopped.
            suffix (Optional[str]): The suffix to append to the prompt.
            temperature (Optional[int]): Controls the randomness of the output.
            top_p (Optional[int]): Controls the cumulative probability of the model's output.
            tools (Optional[List[Dict[str, Union[str, int]]]]): List of tool name, version pairs.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.frequency_penalty = frequency_penalty
        self.functions = functions
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools


class AnthropicConfiguration(ConfigurationBase):
    """
    Configuration class for Anthropic's text generation API.

    Reference: https://docs.anthropic.com/claude/reference/complete_post
    """

    def __init__(
        self,
        max_tokens: Optional[int] = 256,
        max_tokens_to_sample: Optional[int] = 256,
        metadata: Optional[Dict] = None,
        stop_sequences: Optional[List] = None,
        stream: bool = False,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for Anthropic API.

        Args:
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            metadata (Optional[Dict]): Additional metadata to include with the generation request.
            stop_sequences (Optional[List]): List of tokens where generation is stopped.
            system (Optional[str]): Specifies the system to use for text generation.
            temperature (Optional[int]): Controls the randomness of the output.
            top_k (Optional[int]): Controls the diversity of the output by limiting the vocabulary considered.
            top_p (Optional[int]): Controls the cumulative probability of the model's output.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.max_tokens = max_tokens
        self.max_tokens_to_sample = max_tokens_to_sample
        self.metadata = metadata
        self.stop_sequences = stop_sequences
        self.stream = stream
        self.system = system
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class HuggingfaceConfiguration(ConfigurationBase):
    """
    Configuration class for Huggingface's text generation API.

    Reference: https://huggingface.github.io/text-generation-inference/#/Text%20Generation%20Inference/compat_generate
    """

    def __init__(
        self,
        best_of: Optional[int] = None,
        decoder_input_details: Optional[bool] = None,
        details: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        return_full_text: Optional[bool] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_n_tokens: Optional[int] = None,
        top_p: Optional[int] = None,
        truncate: Optional[int] = None,
        typical_p: Optional[float] = None,
        watermark: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """

        This configuration class allows customization of parameters for text generation using Huggingface's text generation API.

        Args:
            best_of (Optional[int]): Number of candidates to sample. Defaults to None.
            decoder_input_details (Optional[bool]): Whether to include decoder input details. Defaults to None.
            details (Optional[bool]): Whether to return additional details. Defaults to None.
            max_new_tokens (Optional[int]): Maximum number of new tokens to generate. Defaults to None.
            repetition_penalty (Optional[float]): Penalty for repetition. Defaults to None.
            return_full_text (Optional[bool]): Whether to return the full text or not. Defaults to None.
            seed (Optional[int]): Seed for random generation. Defaults to None.
            temperature (Optional[float]): Softmax temperature for generation. Defaults to None.
            top_k (Optional[int]): Value for top-k sampling. Defaults to None.
            top_n_tokens (Optional[int]): Number of top tokens to consider. Defaults to None.
            top_p (Optional[int]): Value for top-p sampling. Defaults to None.
            truncate (Optional[int]): Maximum length of generated text. Defaults to None.
            typical_p (Optional[float]): Typical probability for token selection. Defaults to None.
            watermark (Optional[bool]): Whether to include a watermark or not. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base configuration.


        """
        super().__init__(**kwargs)
        self.best_of = best_of
        self.decoder_input_details = decoder_input_details
        self.details = details
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.return_full_text = return_full_text
        self.seed = seed
        self.temperature = temperature
        self.top_k = top_k
        self.top_n_tokens = top_n_tokens
        self.top_p = top_p
        self.truncate = truncate
        self.typical_p = typical_p
        self.watermark = watermark


class GeminiConfiguration(ConfigurationBase):
    """
    Configuration class for Google's Gemini text generation API.

    Reference: https://ai.google.dev/api/python/google/generativeai/GenerationConfig
    """

    def __init__(
        self,
        candidate_count: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> None:
        """

        This configuration class allows customization of parameters for text generation using Google's Gemini API.

        Args:
            candidate_count (Optional[int]): Number of candidates to generate. Defaults to None.
            max_output_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to None.
            stop_sequences (Optional[list]): List of sequences at which to stop generation. Defaults to None.
            temperature (Optional[float]): Softmax temperature for generation. Defaults to None.
            top_k (Optional[int]): Value for top-k sampling. Defaults to None.
            top_p (Optional[float]): Value for top-p sampling. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base configuration.
        """
        super().__init__(**kwargs)
        self.candidate_count = candidate_count
        self.max_output_tokens = max_output_tokens
        self.stop_sequences = stop_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class PalmConfiguration(ConfigurationBase):
    """
    Configuration class for Palm (Generative AI).

    Reference: https://developers.generativeai.google/api/python/google/generativeai/chat
    """

    def __init__(
        self,
        candidate_count: Optional[int] = None,
        context: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> None:
        """

        This configuration class allows customization of parameters for text generation using Palm (Generative AI).

        Args:
            candidate_count (Optional[int]): Number of candidates to generate. Defaults to None.
            context (Optional[str]): Context to provide for generation. Defaults to None.
            max_output_tokens (Optional[int]): Maximum number of tokens to generate. Defaults to None.
            temperature (Optional[float]): Softmax temperature for generation. Defaults to None.
            top_k (Optional[int]): Value for top-k sampling. Defaults to None.
            top_p (Optional[float]): Value for top-p sampling. Defaults to None.
            **kwargs: Additional keyword arguments to be passed to the base configuration.

        """
        super().__init__(**kwargs)
        self.candidate_count = candidate_count
        self.context = context
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class OllamaConfiguration(ConfigurationBase):
    """
    Configuration class for Ollama.

    Reference: https://github.com/jmorganca/ollama/blob/main/docs/api.md#parameters
    """

    def __init__(
        self,
        mirostat: Optional[int] = None,
        mirostat_eta: Optional[float] = None,
        mirostat_tau: Optional[float] = None,
        num_ctx: Optional[int] = None,
        num_gqa: Optional[int] = None,
        num_predict: Optional[int] = None,
        num_thread: Optional[int] = None,
        repeat_last_n: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[list] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        temperature: Optional[float] = None,
        tfs_z: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initializes configuration parameters for Ollama.

        Args:
            mirostat (Optional[int]): Parameter controlling mirostat.
            mirostat_eta (Optional[float]): Parameter controlling mirostat eta.
            mirostat_tau (Optional[float]): Parameter controlling mirostat tau.
            num_ctx (Optional[int]): Number of context tokens.
            num_gqa (Optional[int]): Number of GQA tokens.
            num_predict (Optional[int]): Number of predicted tokens.
            num_thread (Optional[int]): Number of threads.
            repeat_last_n (Optional[int]): Number of repeats.
            repeat_penalty (Optional[float]): Penalty for repeats.
            stop (Optional[list]): List of tokens where generation is stopped.
            system (Optional[str]): System parameter.
            template (Optional[str]): Template parameter.
            temperature (Optional[float]): Temperature parameter.
            tfs_z (Optional[float]): TFS Z parameter.
            top_k (Optional[int]): Top k parameter.
            top_p (Optional[float]): Top p parameter.

            **kwargs: Additional keyword arguments representing configuration parameters.
        """
        super().__init__(**kwargs)
        self.mirostat = mirostat
        self.mirostat_eta = mirostat_eta
        self.mirostat_tau = mirostat_tau
        self.num_ctx = num_ctx
        self.num_gqa = num_gqa
        self.num_predict = num_predict
        self.num_thread = num_thread
        self.repeat_last_n = repeat_last_n
        self.repeat_penalty = repeat_penalty
        self.stop = stop
        self.system = system
        self.template = template
        self.temperature = temperature
        self.tfs_z = tfs_z
        self.top_k = top_k
        self.top_p = top_p


class AI21Configuration(ConfigurationBase):
    """
    Reference: https://docs.ai21.com/reference/j2-complete-ref
    """

    def __init__(
        self,
        countPenalty: Optional[dict] = None,
        frequencePenalty: Optional[dict] = None,
        maxTokens: Optional[int] = None,
        minTokens: Optional[int] = None,
        numResults: Optional[int] = None,
        presencePenalty: Optional[dict] = None,
        stopSequences: Optional[list] = None,
        temperature: Optional[float] = None,
        topKReturn: Optional[int] = None,
        topP: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize AI21Configuration.

        Args:
            countPenalty (Optional[dict]): Count penalty parameters.
            frequencePenalty (Optional[dict]): Frequence penalty parameters.
            maxTokens (Optional[int]): Maximum tokens.
            minTokens (Optional[int]): Minimum tokens.
            numResults (Optional[int]): Number of results.
            presencePenalty (Optional[dict]): Presence penalty parameters.
            stopSequences (Optional[list]): Stop sequences.
            temperature (Optional[float]): Temperature.
            topKReturn (Optional[int]): Top K return.
            topP (Optional[float]): Top P.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.countPenalty = countPenalty
        self.frequencePenalty = frequencePenalty
        self.maxTokens = maxTokens
        self.minTokens = minTokens
        self.numResults = numResults
        self.presencePenalty = presencePenalty
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topKReturn = topKReturn
        self.topP = topP


class AlephAlphaConfiguration(ConfigurationBase):
    """
    Reference: https://docs.aleph-alpha.com/api/complete/
    """

    def __init__(
        self,
        model: Optional[str] = None,
        hosting: Optional[str] = None,
        maximum_tokens: Optional[int] = None,
        minimum_tokens: Optional[int] = None,
        echo: Optional[bool] = False,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        sequence_penalty: Optional[float] = None,
        sequence_penalty_min_length: Optional[int] = 2,
        repetition_penalties_include_prompt: Optional[bool] = False,
        repetition_penalties_include_completion: Optional[bool] = True,
        use_multiplicative_presence_penalty: Optional[bool] = False,
        use_multiplicative_frequency_penalty: Optional[bool] = False,
        use_multiplicative_sequence_penalty: Optional[bool] = False,
        penalty_bias: Optional[str] = None,
        penalty_exceptions: Optional[List[str]] = None,
        penalty_exceptions_include_stop_sequences: Optional[bool] = True,
        best_of: Optional[int] = 2,
        n: Optional[int] = 1,
        logit_bias: Optional[dict] = None,
        stop_sequences: Optional[List[str]] = None,
        tokens: Optional[bool] = False,
        raw_completion: Optional[bool] = False,
        disable_optimizations: Optional[bool] = False,
        completion_bias_inclusion: Optional[List[str]] = [],
        completion_bias_inclusion_first_token_only: Optional[bool] = False,
        completion_bias_exclusion: Optional[List[str]] = [],
        completion_bias_exclusion_first_token_only: Optional[bool] = False,
        contextual_control_threshold: Optional[float] = None,
        control_log_additive: Optional[bool] = True,
    ) -> None:
        """
        Represents a configuration for the model.

        Args:
            model (str, optional): The name of the model from the Luminous model family.
            hosting (str, optional): Possible values: ['aleph-alpha', None].
            maximum_tokens (int, optional): The maximum number of tokens to be generated.
            minimum_tokens (int, optional): Generate at least this number of tokens before an end-of-text token is generated.
            echo (bool, optional): Echo the prompt in the completion.
            temperature (float, optional): A higher sampling temperature encourages the model to produce less probable outputs.
            top_k (int, optional): Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely options.
            top_p (float, optional): Introduces random sampling for generated tokens by randomly selecting the next token from the smallest possible set of tokens whose cumulative probability exceeds the probability top_p.
            presence_penalty (float, optional): The presence penalty reduces the likelihood of generating tokens that are already present in the generated text.
            frequency_penalty (float, optional): The frequency penalty reduces the likelihood of generating tokens that are already present in the generated text.
            sequence_penalty (float, optional): Increasing the sequence penalty reduces the likelihood of reproducing token sequences that already appear in the prompt and prior completion.
            sequence_penalty_min_length (int, optional): Minimal number of tokens to be considered as sequence.
            repetition_penalties_include_prompt (bool, optional): Flag deciding whether presence penalty or frequency penalty are updated from tokens in the prompt.
            repetition_penalties_include_completion (bool, optional): Flag deciding whether presence penalty or frequency penalty are updated from tokens in the completion.
            use_multiplicative_presence_penalty (bool, optional): Flag deciding whether presence penalty is applied multiplicatively or additively.
            use_multiplicative_frequency_penalty (bool, optional): Flag deciding whether frequency penalty is applied multiplicatively or additively.
            use_multiplicative_sequence_penalty (bool, optional): Flag deciding whether sequence penalty is applied multiplicatively or additively.
            penalty_bias (str, optional): All tokens in this text will be used in addition to the already penalized tokens for repetition penalties.
            penalty_exceptions (List[str], optional): List of strings that may be generated without penalty, regardless of other penalty settings.
            penalty_exceptions_include_stop_sequences (bool, optional): Flag deciding whether stop sequences are included in penalty exceptions.
            best_of (int, optional): If a value is given, the number of best_of completions will be generated on the server side.
            n (int, optional): The number of completions to return.
            logit_bias (dict, optional): Log probabilities for each token generated.
            stop_sequences (List[str], optional): List of strings that will stop generation if they're generated.
            tokens (bool, optional): Flag indicating whether individual tokens of the completion should be returned.
            raw_completion (bool, optional): Setting this parameter to true forces the raw completion of the model to be returned.
            disable_optimizations (bool, optional): Flag to disable optimizations applied to prompt and completion.
            completion_bias_inclusion (List[str], optional): Bias the completion to only generate options within this list.
            completion_bias_inclusion_first_token_only (bool, optional): Only consider the first token for the completion bias inclusion.
            completion_bias_exclusion (List[str], optional): Bias the completion to not generate options within this list.
            completion_bias_exclusion_first_token_only (bool, optional): Only consider the first token for the completion bias exclusion.
            contextual_control_threshold (float, optional): Apply control parameters to similar tokens based on similarity score.
            control_log_additive (bool, optional): Apply controls on prompt items by adding the log(control_factor) to attention scores.
        """
        self.model = model
        self.hosting = hosting
        self.maximum_tokens = maximum_tokens
        self.minimum_tokens = minimum_tokens
        self.echo = echo
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.sequence_penalty = sequence_penalty
        self.sequence_penalty_min_length = sequence_penalty_min_length
        self.repetition_penalties_include_prompt = repetition_penalties_include_prompt
        self.repetition_penalties_include_completion = (
            repetition_penalties_include_completion
        )
        self.use_multiplicative_presence_penalty = use_multiplicative_presence_penalty
        self.use_multiplicative_frequency_penalty = use_multiplicative_frequency_penalty
        self.use_multiplicative_sequence_penalty = use_multiplicative_sequence_penalty
        self.penalty_bias = penalty_bias
        self.penalty_exceptions = penalty_exceptions
        self.penalty_exceptions_include_stop_sequences = (
            penalty_exceptions_include_stop_sequences
        )
        self.best_of = best_of
        self.n = n
        self.logit_bias = logit_bias
        self.stop_sequences = stop_sequences
        self.tokens = tokens
        self.raw_completion = raw_completion
        self.disable_optimizations = disable_optimizations
        self.completion_bias_inclusion = completion_bias_inclusion
        self.completion_bias_inclusion_first_token_only = (
            completion_bias_inclusion_first_token_only
        )
        self.completion_bias_exclusion = completion_bias_exclusion
        self.completion_bias_exclusion_first_token_only = (
            completion_bias_exclusion_first_token_only
        )
        self.contextual_control_threshold = contextual_control_threshold
        self.control_log_additive = control_log_additive


class AlephAlphaEmbeddingsConfiguration(ConfigurationBase):
    """
    Reference: https://docs.aleph-alpha.com/api/complete/
    """

    def __init__(
        self,
        hosting: Optional[str] = None,
        layers: Optional[List[int]] = None,
        tokens: Optional[bool] = None,
        pooling: Optional[List[str]] = None,
        type: Optional[str] = None,
        normalize: Optional[bool] = False,
        contextual_control_threshold: Optional[float] = None,
        control_log_additive: Optional[bool] = True,
    ) -> None:
        """
        Initialize configuration attributes with optional parameters.

        Args:
            model (str, optional): Name of the model to use.
            hosting (str, optional): Specifies which datacenters may process the request.
            layers (List[int], optional): List of layer indices from which to return embeddings.
            tokens (bool, optional): Flag indicating whether the tokenized prompt is to be returned.
            pooling (List[str], optional): Pooling operation to use.
            type (str, optional): Explicitly set embedding type to be passed to the model.
            normalize (bool, optional): Flag indicating whether to return normalized embeddings.
            contextual_control_threshold (float, optional): Threshold for contextual control.
            control_log_additive (bool, optional): Flag indicating the method of applying controls.
        """
        self.hosting = hosting
        self.layers = layers
        self.tokens = tokens
        self.pooling = pooling
        self.type = type
        self.normalize = normalize
        self.contextual_control_threshold = contextual_control_threshold
        self.control_log_additive = control_log_additive


class AmazonBedrockTitanConfiguration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-text-express-v1
    """

    def __init__(
        self,
        maxTokenCount: Optional[int] = None,
        stopSequences: Optional[list] = None,
        temperature: Optional[float] = None,
        topP: Optional[int] = None,
    ) -> None:
        """
        Initialize AmazonBedrockTitanConfiguration.

        Args:
            maxTokenCount (Optional[int]): Maximum token count.
            stopSequences (Optional[list]): Stop sequences.
            temperature (Optional[float]): Temperature.
            topP (Optional[int]): Top P.
        """
        super().__init__(
            maxTokenCount=maxTokenCount,
            stopSequences=stopSequences,
            temperature=temperature,
            topP=topP,
        )


class AmazonBedrockAnthropicClaude3Configuration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        anthropic_version: Optional[str] = None,
    ) -> None:
        """
        Initialize AmazonBedrockAnthropicClaude3Configuration.

        Args:
            max_tokens (Optional[int]): Maximum tokens.
            anthropic_version (Optional[str]): Anthropic version.
        """
        super().__init__(max_tokens=max_tokens, anthropic_version=anthropic_version)

    def get_supported_openai_params(self) -> List[str]:
        """
        Get supported OpenAI parameters.

        Returns:
            List[str]: Supported OpenAI parameters.
        """
        return ["max_tokens", "tools", "tool_choice", "stream"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        """
        Map OpenAI parameters.

        Args:
            non_default_params (dict): Non-default parameters.
            optional_params (dict): Optional parameters.

        Returns:
            dict: Mapped OpenAI parameters.
        """
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "stream":
                optional_params["stream"] = value
        return optional_params


class AmazonBedrockAnthropicConfiguration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude
    """

    def __init__(
        self,
        anthropic_version: Optional[str] = None,
        max_tokens_to_sample: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
    ) -> None:
        """
        Initialize AmazonBedrockAnthropicConfiguration.

        Args:
            anthropic_version (Optional[str]): Anthropic version.
            max_tokens_to_sample (Optional[int]): Maximum tokens to sample.
            stop_sequences (Optional[list]): Stop sequences.
            temperature (Optional[float]): Temperature.
            top_k (Optional[int]): Top K.
            top_p (Optional[int]): Top P.

        """
        super().__init__(
            anthropic_version=anthropic_version,
            max_tokens_to_sample=max_tokens_to_sample,
            stop_sequences=stop_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    def get_supported_openai_params(self) -> List[str]:
        """
        Get supported OpenAI parameters.

        Returns:
            List[str]: Supported OpenAI parameters.
        """
        return ["max_tokens", "temperature", "stop", "top_p", "stream"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        """
        Map OpenAI parameters.

        Args:
            non_default_params (dict): Non-default parameters.
            optional_params (dict): Optional parameters.

        Returns:
            dict: Mapped OpenAI parameters.
        """
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens_to_sample"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "stream" and value == True:
                optional_params["stream"] = value
        return optional_params


class AmazonBedrockCohereConfiguration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=command
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_likelihood: Optional[str] = None,
    ) -> None:
        """
        Initialize AmazonBedrockCohereConfiguration.

        Args:
            max_tokens (Optional[int]): Maximum tokens.
            temperature (Optional[float]): Temperature.
            return_likelihood (Optional[str]): Return likelihood.
        """
        super().__init__(
            max_tokens=max_tokens,
            temperature=temperature,
            return_likelihood=return_likelihood,
        )


class AmazonBedrockAI21Configuration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra
    """

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        stopSequences: Optional[list] = None,
        frequencePenalty: Optional[dict] = None,
        presencePenalty: Optional[dict] = None,
        countPenalty: Optional[dict] = None,
    ) -> None:
        """
        Initialize AmazonBedrockAI21Configuration.

        Args:
            maxTokens (Optional[int]): Maximum tokens.
            temperature (Optional[float]): Temperature.
            topP (Optional[float]): Top P.
            stopSequences (Optional[list]): Stop sequences.
            frequencePenalty (Optional[dict]): Frequence penalty.
            presencePenalty (Optional[dict]): Presence penalty.
            countPenalty (Optional[dict]): Count penalty.
        """
        super().__init__(
            maxTokens=maxTokens,
            temperature=temperature,
            topP=topP,
            stopSequences=stopSequences,
            frequencePenalty=frequencePenalty,
            presencePenalty=presencePenalty,
            countPenalty=countPenalty,
        )


class AmazonBedrockLlamaConfiguration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=meta.llama2-13b-chat-v1
    """

    def __init__(
        self,
        max_gen_len: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
    ) -> None:
        """
        Initialize AmazonBedrockLlamaConfiguration.

        Args:
            max_gen_len (Optional[int]): Maximum generation length.
            temperature (Optional[float]): Temperature.
            topP (Optional[float]): Top P.
        """
        super().__init__(max_gen_len=max_gen_len, temperature=temperature, topP=topP)


class AmazonBedrockMistralConfiguration(ConfigurationBase):
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        top_k: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize AmazonBedrockMistralConfiguration.

        Args:
            max_tokens (Optional[int]): Maximum tokens.
            temperature (Optional[float]): Temperature.
            top_p (Optional[int]): Top P.
            top_k (Optional[float]): Top K.
            stop (Optional[List[str]]): Stop sequences.
        """
        super().__init__(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )


class AmazonBedrockStabilityConfiguration(ConfigurationBase):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0
    """

    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
    ) -> None:
        """
        Initialize AmazonBedrockStabilityConfiguration.

        Args:
            cfg_scale (Optional[int]): Configuration scale.
            height (Optional[int]): Height.
            seed (Optional[float]): Seed.
            steps (Optional[List[str]]): Steps.
            width (Optional[int]): Width.
        """
        super().__init__(
            cfg_scale=cfg_scale,
            height=height,
            seed=seed,
            steps=steps,
            width=width,
        )


class CohereConfiguration(ConfigurationBase):
    def __init__(
        self,
        chat_history: Optional[List] = None,
        connectors: Optional[List] = None,
        conversation_id: Optional[str] = None,
        documents: Optional[List] = None,
        frequency_penalty: Optional[int] = None,
        generation_id: Optional[str] = None,
        k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        preamble: Optional[str] = None,
        presence_penalty: Optional[int] = None,
        prompt_truncation: Optional[str] = None,
        p: Optional[int] = None,
        response_id: Optional[str] = None,
        search_queries_only: Optional[bool] = None,
        temperature: Optional[float] = None,
        tool_results: Optional[List] = None,
        tools: Optional[List] = None,
        **kwargs,
    ) -> None:
        """
        Initialize CohereConfiguration.

        Args:
            chat_history (Optional[List]): Chat history.
            connectors (Optional[List]): Connectors.
            conversation_id (Optional[str]): Conversation ID.
            documents (Optional[List]): Documents.
            frequency_penalty (Optional[int]): Frequency penalty.
            generation_id (Optional[str]): Generation ID.
            k (Optional[int]): K.
            max_tokens (Optional[int]): Maximum tokens.
            preamble (Optional[str]): Preamble.
            presence_penalty (Optional[int]): Presence penalty.
            prompt_truncation (Optional[str]): Prompt truncation.
            p (Optional[int]): P.
            response_id (Optional[str]): Response ID.
            search_queries_only (Optional[bool]): Search queries only.
            temperature (Optional[int]): Temperature.
            tool_results (Optional[List]): Tool results.
            tools (Optional[List]): Tools.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.chat_history = chat_history
        self.connectors = connectors
        self.conversation_id = conversation_id
        self.documents = documents
        self.frequency_penalty = frequency_penalty
        self.generation_id = generation_id
        self.k = k
        self.max_tokens = max_tokens
        self.preamble = preamble
        self.presence_penalty = presence_penalty
        self.prompt_truncation = prompt_truncation
        self.p = p
        self.response_id = response_id
        self.search_queries_only = search_queries_only
        self.temperature = temperature
        self.tool_results = tool_results
        self.tools = tools


class CohereEmbeddingConfiguration(ConfigurationBase):

    def __init__(
        self,
        input_type=None,
        embedding_types=None,
        truncate=None,
    ):
        self.input_type = input_type
        self.embedding_types = embedding_types
        self.truncate = truncate


class SagemakerConfiguration(ConfigurationBase):

    def __init__(
        self,
        max_new_tokens: Optional[List] = None,
        return_full_text: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.max_new_tokens = max_new_tokens
        self.return_full_text = return_full_text
        self.temperature = temperature
        self.top_p = top_p


class VertexConfiguration(ConfigurationBase):

    def __init__(
        self,
        max_output_tokens: Optional[List] = None,
        temperature: Optional[List] = None,
        top_k: Optional[float] = None,
        top_p: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class TogetherAIConfiguration(ConfigurationBase):

    def __init__(
        self,
        logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        stop: Optional[str] = None,
        temperature: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)
        self.logprobs = logprobs
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.stop = stop
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class BedrockConfigFactory(ConfigurationBase):
    def __init__(self):
        self.bedrock_config_mapping = {
            "ai21": AmazonBedrockAI21Configuration,
            "anthropic": AmazonBedrockAnthropicConfiguration,
            "claude3": AmazonBedrockAnthropicClaude3Configuration,
            "cohere": AmazonBedrockCohereConfiguration,
            "llama": AmazonBedrockLlamaConfiguration,
            "mistral": AmazonBedrockMistralConfiguration,
            "stability": AmazonBedrockStabilityConfiguration,
            "titan": AmazonBedrockTitanConfiguration,
        }

    def get_provider(self, provider):
        if provider in self.bedrock_config_mapping:
            return self.bedrock_config_mapping[provider]
        else:
            raise ValueError(
                f"No configuration found for the P{provider} for AWS Bedrock "
            )


class LLMConfiguration(ConfigurationBase):
    """
    Consolidated configuration class for various for all supported providers.
    """

    def __init__(
        self,
        ai21api_config: Optional[AI21Configuration] = None,
        aleph_alpha_config: Optional[AlephAlphaConfiguration] = None,
        anthropic_api_config: Optional[AnthropicConfiguration] = None,
        bedrock_config: Optional[BedrockConfigFactory] = None,
        cohere_config: Optional[CohereConfiguration] = None,
        gemini_config: Optional[GeminiConfiguration] = None,
        huggingface_config: Optional[HuggingfaceConfiguration] = None,
        ollama_config: Optional[OllamaConfiguration] = None,
        openai_config: Optional[OpenAIConfiguration] = None,
        palm_config: Optional[PalmConfiguration] = None,
        sagemaker_config: Optional[SagemakerConfiguration] = None,
        together_config: Optional[TogetherAIConfiguration] = None,
        vertex_config: Optional[VertexConfiguration] = None,
        **kwargs,
    ) -> None:
        """
        Initialize LLMConfiguration.

        Args:
            ai21api_config (Optional[AI21Configuration]): AI21 API configuration.
            aleph_alpha_config (Optional[AlephAlphaConfiguration]): Aleph Alpha configuration.
            anthropic_api_config (Optional[AnthropicConfiguration]): Anthropic API configuration.
            bedrock_config (Optional[BedrockConfigFactory]): Amazon Bedrock configuration.
            cohere_config (Optional[CohereConfiguration]): Cohere configuration.
            gemini_config (Optional[GeminiConfiguration]): Gemini configuration.
            huggingface_config (Optional[HuggingfaceConfiguration]): Huggingface configuration.
            ollama_config (Optional[OllamaConfiguration]): Ollama configuration.
            openai_config (Optional[OpenAIConfiguration]): OpenAI configuration.
            palm_config (Optional[PalmConfiguration]): Palm configuration.
            sagemaker_config (Optional[SagemakerConfiguration]): Sagemaker configuration.
            together_config (Optional[SagemakerConfiguration]): Sagemaker configuration.
            vertex_config (Optional[VertexConfiguration]): Vertex configuration.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.ai21api_config = ai21api_config
        self.aleph_alpha_config = aleph_alpha_config
        self.anthropic_api_config = anthropic_api_config
        self.bedrock_config = bedrock_config
        self.cohere_config = cohere_config
        self.gemini_config = gemini_config
        self.huggingface_config = huggingface_config
        self.ollama_config = ollama_config
        self.openai_config = openai_config
        self.palm_config = palm_config
        self.sagemaker_config = sagemaker_config
        self.together_config = together_config
        self.vertex_config = vertex_config

    def config(self) -> Dict:
        """
        Returns a dictionary representation of the configuration.

        Returns:
            Dict: Dictionary containing non-None instance variables.
        """
        config_dict = super().config()
        for config_name, config_obj in self.__dict__.items():
            if isinstance(config_obj, ConfigurationBase):
                config_dict[config_name] = config_obj.config()
        return config_dict
