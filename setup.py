import io
import platform
from os import path

from setuptools import find_packages, setup

# Baca README.md untuk long description
with io.open("README.md", encoding="utf-8") as f:
    readme = f.read()

# Baca requirements.txt
info = path.abspath(path.dirname(__file__))
with io.open(path.join(info, "requirements.txt"), encoding="utf-8") as f:
    core_require = f.read().splitlines()

install_require = [
    x.strip() for x in core_require if x.strip() and not x.startswith("git+")
]

setup(
    name="Oculus",
    version="0.0.1",
    description="Library untuk deteksi objek",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_require,
    license="MIT",
)
