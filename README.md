# Amazon Braket Algorithm Library
[![Build](https://github.com/aws-samples/amazon-braket-algorithm-library/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/aws-samples/amazon-braket-algorithm-library/actions/workflows/python-package.yml)

The Braket Algorithm Library provides Amazon Braket customers with pre-built implementations of prominent quantum algorithms and experimental workloads as ready-to-run example notebooks.

---
Currently, Braket algorithms are tested on Linux, Windows, and Mac.

Running notebooks locally requires additional dependencies located in [notebooks/textbook/requirements.txt](https://github.com/aws-samples/amazon-braket-algorithm-library/blob/main/notebooks/textbook/requirements.txt). See notebooks/textbook/README.md for more information.

---
## <a name="install">Installing the Amazon Braket Algorithm Library</a>
The Amazon Braket Algorithm Library can be installed from source by cloning this repository and running a pip install command in the root directory of the repository.

```bash
git clone https://github.com/aws-samples/amazon-braket-algorithm-library.git
cd amazon-braket-algorithm-library
pip install .
```

To run the notebook examples locally on your IDE, first, configure a profile to use your account to interact with AWS. To learn more, see [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

After you create a profile, use the following command to set the `AWS_PROFILE` so that all future commands can access your AWS account and resources.

```bash
export AWS_PROFILE=YOUR_PROFILE_NAME
```

### Configure your AWS account with the resources necessary for Amazon Braket
If you are new to Amazon Braket, onboard to the service and create the resources necessary to use Amazon Braket using the [AWS console](https://console.aws.amazon.com/braket/home ).


## Support

### Issues and Bug Reports

If you encounter bugs or face issues while using the algorithm library, please let us know by posting 
the issue on our [Github issue tracker](https://github.com/aws-samples/amazon-braket-algorithm-library/issues).  
For other issues or general questions, please ask on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/questions/ask) and add the tag amazon-braket.

### Feedback and Feature Requests

If you have feedback or features that you would like to see on Amazon Braket, we would love to hear from you!  
[Github issues](https://github.com/aws-samples/amazon-braket-algorithm-library/issues) is our preferred mechanism for collecting feedback and feature requests, allowing other users 
to engage in the conversation, and +1 issues to help drive priority. 


## License
This project is licensed under the Apache-2.0 License.
