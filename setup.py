"""Set-up file for the package."""

import os
import pathlib
from typing import List

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version() -> str:
    """Gets the JAX-EC version."""
    path = CWD / "jax_ec" / "__init__.py"
    content = path.read_text()
    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise ValueError("Version not found in jax_ec/__init__.py")


def _parse_requirements(path: str) -> List[str]:
    """Returns the requirements from the file at the given path."""
    with open(os.path.join(path)) as f:
        reqs = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Convert == to >= for library compatibility
            line = line.replace("==", ">=")
            reqs.append(line)
        return reqs


print(get_version())
setup(
    name="jax_ec",
    install_requires=_parse_requirements("requirements/requirements.txt"),
    # extras_require={
    #     "dev": _parse_requirements("requirements/requirements-dev.txt"),
    # },
    version=get_version(),
    long_description=open("README.md").read(),
)
