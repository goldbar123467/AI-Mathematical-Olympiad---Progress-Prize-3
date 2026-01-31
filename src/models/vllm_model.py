"""vLLM model wrapper for high-throughput inference."""

from typing import List, Dict, Any


class VLLMModel:
    """Wrapper for vLLM inference engine."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-72B-Instruct",
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int = 8192
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.llm = None
        self.tokenizer = None

    def load(self):
        """Load the model using vLLM."""
        from vllm import LLM

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=True
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 4096,
        top_p: float = 0.95,
        stop: List[str] = None
    ) -> str:
        """Generate response from chat messages."""
        from vllm import SamplingParams

        if self.llm is None:
            self.load()

        # Format messages for chat model
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or []
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[str]:
        """Generate responses for a batch of message lists."""
        from vllm import SamplingParams

        if self.llm is None:
            self.load()

        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.1),
            top_p=kwargs.get('top_p', 0.95),
            max_tokens=kwargs.get('max_tokens', 4096),
            stop=kwargs.get('stop', [])
        )

        outputs = self.llm.generate(prompts, sampling_params)
        return [output.outputs[0].text for output in outputs]
