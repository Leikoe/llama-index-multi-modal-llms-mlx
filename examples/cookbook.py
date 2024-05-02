# from https://docs.llamaindex.ai/en/stable/examples/multi_modal/ollama_cookbook/
from llama_index.multi_modal_llms.mlx import MlxMultiModal

llm = MlxMultiModal(model="mlx-community/llava-llama-3-8b-v1_1-4bit")
print("model loaded")

from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from PIL import Image
import matplotlib.pyplot as plt

image_documents = SimpleDirectoryReader("./assets/restaurant_images").load_data()


from pydantic import BaseModel


class Restaurant(BaseModel):
    """Data model for an restaurant."""

    restaurant: str
    food: str
    discount: str
    discount_reasons: str
    discount_conditions: str
    price: str
    rating: str
    review: str

from llama_index.core.program import MultiModalLLMCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser

prompt_template_str = """\
{query_str}

Return the answer as a Pydantic object. The Pydantic schema is given below:

"""
mm_program = MultiModalLLMCompletionProgram.from_defaults(
    output_parser=PydanticOutputParser(Restaurant),
    image_documents=image_documents,
    prompt_template_str=prompt_template_str,
    multi_modal_llm=llm,
    verbose=True,
)

response = mm_program(query_str="Can you summarize what is in the image?")
for res in response:
    print(res)
