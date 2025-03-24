#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import io
from setuptools import setup, find_packages

# Package meta-data
NAME = 'renaissance_ocr'
DESCRIPTION = 'OCR system for Renaissance documents using quantized vision-language models and agentic architecture'
URL = 'https://github.com/DebjyotiRay/renaissance-ocr'
EMAIL = 'debjyotiray0104@gmail.com'
AUTHOR = 'Debjyoti Ray'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.1.0'

# Required packages
REQUIRED = [
    'torch>=2.0.0',
    'torchvision>=0.15.0',
    'transformers>=4.35.0',
    'bitsandbytes>=0.41.0',
    'accelerate>=0.23.0',
    'peft>=0.6.0',
    'pillow>=10.0.0',
    'opencv-python>=4.8.0',
    'numpy>=1.24.0',
    'matplotlib>=3.7.0',
    'scikit-image>=0.20.0',
    'pandas>=2.0.0',
    'tqdm>=4.65.0',
    'langchain>=0.0.300',
    'autogen>=0.2.0',
    'huggingface_hub>=0.17.0',
    'pytesseract>=0.3.10',
    'difflib>=0.1',
    'PyPDF2>=3.0.0',
    'python-Levenshtein>=0.20.0',
    'reportlab>=3.6.0',
]

# Optional packages
EXTRAS = {
    'dev': [
        'pytest>=7.0.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'isort>=5.12.0',
        'mypy>=1.0.0',
    ],
    'docs': [
        'sphinx>=6.0.0',
        'sphinx-rtd-theme>=1.2.0',
        'sphinx-autodoc-typehints>=1.23.0',
    ],
}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION

setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Text Processing',
    ],
    entry_points={
        'console_scripts': [
            'renaissance-ocr=renaissance_ocr_cli:main',
        ],
    },
)