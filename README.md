# Amazon Braket Algorithm Library
The Braket Algorithm Library provides Amazon Braket customers with pre-built implementations of prominent quantum algorithms and experimental workloads as ready-to-run example notebooks.

---
Currently, Braket algorithms are tested on Linux, Windows, and Mac.

Running notebooks locally requires additional dependencies located in [notebooks/textbook/requirements.txt](https://github.com/aws-samples/amazon-braket-algorithm-library/blob/main/notebooks/textbook/requirements.txt). See notebooks/textbook/README.md for more information.

---
## <a name="conda">Creating a conda environment</a>
To install the dependencies required for running the notebook examples in this repository you can create a conda environment with below commands.

```bash
conda env create -n <your_env_name> -f environment.yml
```

Activate the conda environment using:
```bash
conda activate <your_env_name>
```

To remove the conda environment use:
```bash
conda deactivate
```

For more information, please see [conda usage](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

To run the notebook examples locally on your IDE, first, configure a profile to use your account to interact with AWS. To learn more, see [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html).

After you create a profile, use the following command to set the `AWS_PROFILE` so that all future commands can access your AWS account and resources.

```bash
export AWS_PROFILE=YOUR_PROFILE_NAME
```

### Configure your AWS account with the resources necessary for Amazon Braket
If you are new to Amazon Braket, onboard to the service and create the resources necessary to use Amazon Braket using the [AWS console](https://console.aws.amazon.com/braket/home ).



## License
This project is licensed under the Apache-2.0 License.
