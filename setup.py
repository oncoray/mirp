from setuptools import setup

with open("VERSION.txt", "R") as version_file:
    version = version_file.read().strip()

setup(
    name="mirp",
    version=version,
    description="Medical Image Radiomics Processor",
    url="https://github.com/oncoray/mirp",
    license="EUPL1.2",
    author="Alex Zwanenburg",
    packages=["mirp", "mirp.featureSets", "mirp.imageFilters"],
)