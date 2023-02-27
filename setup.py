# Copyright Amazon.com Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from setuptools import find_namespace_packages, setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("src/braket/_algos/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="amazon-braket-algorithm-library",
    version=version,
    license="Apache License 2.0",
    python_requires=">= 3.7.2",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "amazon-braket-sdk>=1.35.1",
        "scipy>=1.5.2",
        "pennylane>=0.25.1,<=0.27.0",
        "openfermion==1.0.0",
    ],
    extras_require={
        "test": [
            "black>=22.3.0",
            "flake8<=5.0.4",
            "isort",
            "pytest",
            "pytest-cov",
            "pytest-rerunfailures",
            "pytest-xdist",
            "sphinx",
            "sphinx-rtd-theme",
            "sphinxcontrib-apidoc",
            "tox",
        ]
    },
    include_package_data=True,
    url="https://github.com/aws-samples/amazon-braket-algorithm-library",
    author="Amazon Web Services",
    description=(
        "An open source library of quantum computing algorithms implemented on Amazon Braket"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Amazon AWS Quantum",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
    ],
)
