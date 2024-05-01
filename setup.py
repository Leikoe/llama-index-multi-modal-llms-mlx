import sys
from pathlib import Path

from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir / "llama_index_multi_modal_llms_mlx"
sys.path.append(str(package_dir))

# Read the requirements from the requirements.txt file
requirements_path = root_dir / "requirements.txt"
with open(requirements_path) as fid:
    requirements = [l.strip() for l in fid.readlines()]

# Setup configuration
setup(
    name="llama-index-multi-modal-llms-mlx",
    version="0.1.0",
    description="mlx-vlm MultiModalLLM integration for llamaindex",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Leikoe",
    url="https://github.com/Leikoe/llama-index-multi-modal-llms-mlx",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(where=root_dir),
    python_requires=">=3.8",
)
