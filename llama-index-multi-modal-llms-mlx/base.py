import logging
from typing import Any, Callable, Dict, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
)
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.constants import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from llama_index.core.base.llms.generic_utils import (
    messages_to_prompt as generic_messages_to_prompt,
)
from llama_index.core.multi_modal_llms import (
    MultiModalLLM,
    MultiModalLLMMetadata,
)
from llama_index.core.schema import ImageDocument
import codecs
from mlx_vlm.utils import generate, get_model_path, load, load_config, load_image_processor

_logger = logging.getLogger(__name__)

def get_model_and_processors(model_path):
    model_path = get_model_path(model_path)
    config = load_config(model_path)
    model, processor = load(model_path, {"trust_remote_code": True})
    image_processor = load_image_processor(config)
    return model, processor, image_processor


class MlxMultiModal(MultiModalLLM):
    model: str = Field(description="The Multi-Modal model to use from mlx-vlm.")
    temperature: float = Field(
        description="The temperature to use for sampling. Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic."
    )
    max_new_tokens: int = Field(
        description=" The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt"
    )
    # context_window: int = Field(
    #     description="The maximum number of context tokens for the model."
    # )
    top_p: float = Field(
        description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens."
    )
    # num_beams: int = Field(description="Number of beams for beam search decoding.")
    repetition_penalty: float = Field(
        description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it."
    )

    _messages_to_prompt: Callable = PrivateAttr()
    _completion_to_prompt: Callable = PrivateAttr()
    _model = PrivateAttr()
    _processor = PrivateAttr()
    _image_processor = PrivateAttr()

    def __init__(
        self,
        model: str = "mlx-community/llava-llama-3-8b-v1_1-4bit",
        temperature: float = 0.75,
        max_new_tokens: int = 512,
        num_input_files: int = 1,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        repetition_penalty: Optional[float] = 1.0,
        num_beams: Optional[int] = 1,
        top_p: Optional[float] = 0.9,
        messages_to_prompt: Optional[Callable] = None,
        completion_to_prompt: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        self._messages_to_prompt = messages_to_prompt or generic_messages_to_prompt
        self._completion_to_prompt = completion_to_prompt or (lambda x: x)

        self._model, self._processor, self._image_processor = get_model_and_processors(model)

        super().__init__(
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_input_files=num_input_files,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
        )

    @classmethod
    def class_name(cls) -> str:
        return "mlx_vlm_multi_modal_llm"

    @property
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi Modal LLM metadata."""
        return MultiModalLLMMetadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
        )

    def complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        if len(image_documents) > 1:
            _logger.warning(
                "ReplicateMultiModal currently only supports uploading one image document"
                "we are using the first image document for completion."
            )
        if image_documents[0].image_path is None:
            raise ValueError("Image path is required for mlx-vlm completion")

        prompt = self._completion_to_prompt(prompt)
        prompt = codecs.decode(prompt, "unicode_escape")

        if "chat_template" in self._processor.__dict__.keys():
            prompt = self._processor.apply_chat_template(
                [{"role": "user", "content": f"<image>\n{prompt}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "tokenizer" in self._processor.__dict__.keys():
            prompt = self._processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<image>\n{prompt}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            ValueError(
                "Error: processor does not have 'chat_template' or 'tokenizer' attribute."
            )


        return CompletionResponse(text=generate(
            model=self._model,
            processor=self._processor,
            image=image_documents[0].image_path,
            prompt=prompt,
            image_processor=self._image_processor,
            temp=self.temperature,
            max_tokens=self.max_new_tokens,
            verbose=False,
            repetition_penalty=None,
            top_p=self.top_p,
        ))

    def stream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseGen:
        raise NotImplementedError

    # ===== Async Endpoints =====

    async def acomplete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, image_documents, **kwargs)

    async def astream_complete(
        self, prompt: str, image_documents: Sequence[ImageDocument], **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponse:
        raise NotImplementedError

    async def astream_chat(
        self,
        messages: Sequence[ChatMessage],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError
