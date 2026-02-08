from setuptools import setup, find_packages

setup(
    name="smell-to-molecule",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        # ... other dependencies
    ],
    author="Your Name",
    description="NeoBERT for Smell-to-Text Translation",
    python_requires='>=3.8',
)
