#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from file
def read_requirements(filename):
    """Read requirements from a requirements file"""
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

install_requires = read_requirements('requirements.txt')
dev_requires = read_requirements('requirements-dev.txt')

setup(
    name="tiny-llm-edge-profiler",
    version="0.1.0",
    author="Your Organization",
    author_email="contact@your-org.com",
    description="Comprehensive profiling toolkit for running quantized LLMs on edge devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/tiny-llm-edge-profiler",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/tiny-llm-edge-profiler/issues",
        "Documentation": "https://docs.your-org.com/tiny-llm-profiler",
        "Source Code": "https://github.com/your-org/tiny-llm-edge-profiler",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Embedded Systems",
        "Topic :: System :: Hardware",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme",
            "myst-parser",
            "sphinx-autodoc-typehints",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-xdist",
            "hypothesis",
            "pytest-mock",
        ],
        "hardware": [
            "pyserial>=3.5",
            "pyusb>=1.2.1",
            "smbus2>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tiny-profiler=tiny_llm_profiler.cli:main",
            "edge-profiler=tiny_llm_profiler.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tiny_llm_profiler": [
            "firmware/*.bin",
            "configs/*.yaml",
            "templates/*.html",
            "schemas/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "edge-ai", "llm", "quantization", "profiling", "microcontroller",
        "esp32", "stm32", "risc-v", "embedded", "performance", "optimization",
        "tinyml", "iot", "machine-learning", "inference"
    ],
)