from setuptools import setup, find_packages  # type: ignore

with open("README.md") as f:
    long_description = f.read()

setup(
    name="followthemoney-compare",
    version="0.3.3",
    author="Organized Crime and Corruption Reporting Project",
    author_email="data@occrp.org",
    url="https://github.com/alephdata/followthemoney-compare/",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(
        exclude=[
            "tests",
            "cache",
        ]
    ),
    namespace_packages=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "click >= 7.0",
        "numpy >= 1.20.0",
        "pandas >= 1.2.2",
        "followthemoney >= 2.5.0",
        "tqdm >= 4.50.0",
        "mmh3",
    ],
    extras_require={
        "dev": [
            "pymc3 >= 3.11.2",
            "scikit-learn >= 0.24.1",
            "matplotlib >= 3.4.1",
            "seaborn >= 0.11.1",
            "arviz >= 0.11.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "ftm-compare = followthemoney_compare.cli:main",
            "followthemoney-compare = followthemoney_compare.cli:main",
        ],
    },
    tests_require=["coverage", "nose"],
)
