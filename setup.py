import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="data_fast_insights",
    version="0.4.4",
    author="p.gafiatullin; a.tolmachev",
    author_email="p.gafiatullin@xsolla.com",
    description="Analytical module for getting quick insights from data based on central tendency statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ksetrae/data-fast-insights",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
