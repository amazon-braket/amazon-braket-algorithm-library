# Contributing Guidelines

Thank you for your interest in contributing to our project. Whether it's a bug report, new feature, correction, or additional
documentation, we greatly value feedback and contributions from our community.

Please read through this document before submitting any issues or pull requests to ensure we have all the necessary
information to effectively respond to your bug report or contribution.


## Reporting Bugs/Feature Requests

We welcome you to use the [GitHub issue tracker](https://github.com/aws-samples/amazon-braket-algorithm-library/issues) to report bugs or suggest features.

When filing an issue, please check existing open, or recently closed, issues to make sure somebody else hasn't already
reported the issue. Please try to include as much information as you can. Details like these are incredibly useful:

* A reproducible test case or series of steps
* The version of our code being used
* Any modifications you've made relevant to the bug
* Anything unusual about your environment or deployment


## Contributing via Pull Requests
Contributions via pull requests are much appreciated. Before sending us a pull request, please ensure that:

1. You are working against the latest source on the *main* branch.
2. You check existing open, and recently merged, pull requests to make sure someone else hasn't addressed the problem already.
3. You open an issue to discuss any significant work - we would hate for your time to be wasted.

To send us a pull request, please:

1. Fork the repository.
2. Modify the source; please focus on the specific change you are contributing. If you also reformat all the code, it will be hard for us to focus on your change.
3. Ensure local tests pass.
4. Commit to your fork using clear commit messages.
5. Send us a pull request, answering any default questions in the pull request interface.
6. Pay attention to any automated CI failures reported in the pull request, and stay involved in the conversation.

GitHub provides additional documentation on [forking a repository](https://help.github.com/articles/fork-a-repo/) and
[creating a pull request](https://help.github.com/articles/creating-a-pull-request/).


### Making your changes

When you make a contribution please ensure that you:

1.  Follow the existing flow of an example algorithm. This should include providing a circuit definition function a run_<name_of_algo>() function, and a get_<name_of_algo>_results() function
2.  Provide the following files:
    1.  *src/braket/experimental/algorithms/<name_of_algo>/<name_of_algo>.py* - implements your example algorithm
    2.  *src/braket/experimental/algorithms/<name_of_algo>/\_\_init__\.py* - used for testing/packaging
    3.  *notebooks/textbook/<Algorithm_Name>_Algorithm.ipynb* - provides a notebook the runs an example using your implementation
    4.  (optional)*test/unit_tests/braket/experimental/algorithms/<name_of_algo>/test_<name_of_algo>.py* - unit tests for your python file
3.  Only have Open Source licensed dependencies in your example.
4.  Run your algorithm on a simulator and optionally on a QPU in your notebook.
5.  Ensure that your example runs without issues on both a recent Braket Notebook Instance (create a new Braket Notebook Instance or restart one from Amazon Braket [in the console](https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-create-notebook.html)) and locally, using our most [recently released Amazon Braket SDK version](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md#installing-the-amazon-braket-python-sdk). Run the entire notebook by clicking `Cells > Run All`, either in the console or locally, and confirm that every cell completes without error.

In addition we encourage re-use of existing examples but it is not required. If you see an opportunity to make use of existing modules,
feel free to do so. For instance if your example implementation requires Quantum Fourier Transform and you can use the existing
Quantum Fourier Transform module instead of re-implementing it, please do so.


## Finding contributions to work on
The goal of the algorithm library is to offer example implementations of quantum algorithms on Amazon Braket, ranging from textbook algorithms
to advanced implementations of recently published research. If you just read a research paper on an algorithm it may be a good candidate.
Also looking at the existing issues is a great way to find something to contribute on. As our projects, by default, use the default GitHub
issue labels (enhancement/bug/duplicate/help wanted/invalid/question/wontfix), looking at any 'help wanted' issues is a great place to start.


## Code of Conduct
This project has adopted the [Amazon Open Source Code of Conduct](https://aws.github.io/code-of-conduct).
For more information see the [Code of Conduct FAQ](https://aws.github.io/code-of-conduct-faq) or contact
opensource-codeofconduct@amazon.com with any additional questions or comments.


## Security issue notifications
If you discover a potential security issue in this project we ask that you notify AWS/Amazon Security via our [vulnerability reporting page](http://aws.amazon.com/security/vulnerability-reporting/). Please do **not** create a public github issue.


## Licensing

See the [LICENSE](LICENSE) file for our project's licensing. We will ask you to confirm the licensing of your contribution.
