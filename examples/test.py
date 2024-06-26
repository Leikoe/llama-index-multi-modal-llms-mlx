from llama_index.core.schema import ImageDocument
from llama_index.multi_modal_llms.mlx import MlxMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls

llm = MlxMultiModal(model="mlx-community/llava-llama-3-8b-v1_1-4bit")
completion = llm.complete("Describe this image", [ImageDocument(image_path="./assets/demo-2.jpg")])
print(completion.text)
