from setuptools import setup, find_packages
import os

# Get the directory of this setup.py file
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymesh3d",
    version="0.1.0",
    author="Matias Nielsen",
    author_email="matiasnhmb@gmail.com",
    description="Transformer Library for 3D Mesh Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MatN23/PyMesh3D",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "full": [
            "wandb>=0.15.0",
            "matplotlib>=3.5.0",
            "scipy>=1.9.0",
            "scikit-learn>=1.1.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    keywords="3d mesh transformer deep-learning pytorch computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/MatN23/pymesh3d/issues",
        "Source": "https://github.com/MatN23/pymesh3d",
    },
)