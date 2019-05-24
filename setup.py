import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hate-cl-marcorlk",
    version="0.0.8",
    author="Marco Kuroiva and Lucas Malta",
    author_email="marco.antoniorl10@gmail.com",
    description="A packaging test for our hate speech classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tcc-lucas-marco/hate_cl",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)