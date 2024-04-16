# Amazon Braket Algorithm Library
[![Build](https://github.com/amazon-braket/amazon-braket-algorithm-library/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/amazon-braket/amazon-braket-algorithm-library/actions/workflows/python-package.yml)

The Braket Algorithm Library provides Amazon Braket customers with pre-built implementations of prominent quantum algorithms and experimental workloads as ready-to-run example notebooks.

---
### Braket algorithms

Currently, Braket algorithms are tested on Linux, Windows, and Mac.

Running notebooks locally requires additional dependencies located in [notebooks/textbook/requirements.txt](https://github.com/amazon-braket/amazon-braket-algorithm-library/blob/main/notebooks/textbook/requirements.txt). See notebooks/textbook/README.md for more information.

| Textbook algorithms | Notebook | References | 
| ----- | ----- | ----- |
| Bell's Inequality     | [Bells_Inequality.ipynb](notebooks/textbook/Bells_Inequality.ipynb)     | [Bell1964](https://journals.aps.org/ppf/abstract/10.1103/PhysicsPhysiqueFizika.1.195), [Greenberger1990](https://doi.org/10.1119/1.16243)     |
| Bernstein–Vazirani | [Bernstein_Vazirani_Algorithm.ipynb](notebooks/textbook/Bernstein_Vazirani_Algorithm.ipynb) | [Bernstein1997](https://epubs.siam.org/doi/10.1137/S0097539796300921) |
| CHSH Inequality | [CHSH_Inequality.ipynb](notebooks/textbook/CHSH_Inequality.ipynb) | [Clauser1970](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.23.880) |
| Deutsch-Jozsa | [Deutsch_Jozsa_Algorithm.ipynb](notebooks/textbook/Deutsch_Jozsa_Algorithm.ipynb) | [Deutsch1992](https://royalsocietypublishing.org/doi/10.1098/rspa.1992.0167) |
| Grover's Search | [Grovers_Search.ipynb](notebooks/textbook/Grovers_Search.ipynb) | [Figgatt2017](https://www.nature.com/articles/s41467-017-01904-7), [Baker2019](https://arxiv.org/abs/1904.01671) |
| QAOA | [Quantum_Approximate_Optimization_Algorithm.ipynb](notebooks/textbook/Quantum_Approximate_Optimization_Algorithm.ipynb) | [Farhi2014](https://arxiv.org/abs/1411.4028) |
| Quantum Circuit Born Machine | [Quantum_Circuit_Born_Machine.ipynb](notebooks/textbook/Quantum_Circuit_Born_Machine.ipynb) | [Benedetti2019](https://www.nature.com/articles/s41534-019-0157-8),  [Liu2018](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.062324) | 
| QFT | [Quantum_Fourier_Transform.ipynb](notebooks/textbook/Quantum_Fourier_Transform.ipynb) | [Coppersmith2002](https://arxiv.org/abs/quant-ph/0201067) | 
| QPE | [Quantum_Phase_Estimation_Algorithm.ipynb](notebooks/textbook/Quantum_Phase_Estimation_Algorithm.ipynb) | [Kitaev1995](https://arxiv.org/abs/quant-ph/9511026) |
| Quantum Walk | [Quantum_Walk.ipynb](notebooks/textbook/Quantum_Walk.ipynb) | [Childs2002](https://arxiv.org/abs/quant-ph/0209131) |
|Shor's| [Shors_Algorithm.ipynb](notebooks/textbook/Shors_Algorithm.ipynb) | [Shor1998](https://arxiv.org/abs/quant-ph/9508027) |
| Simon's | [Simons_Algorithm.ipynb](notebooks/textbook/Simons_Algorithm.ipynb) | [Simon1997](https://epubs.siam.org/doi/10.1137/S0097539796298637) |


| Advanced algorithms | Notebook | References |
| ----- | ----- | ----- |
| Quantum PCA | [Quantum_Principal_Component_Analysis.ipynb](notebooks/advanced_algorithms/Quantum_Principal_Component_Analysis.ipynb) | [He2022](https://ieeexplore.ieee.org/document/9669030) |
| QMC | [Quantum_Computing_Quantum_Monte_Carlo.ipynb](notebooks/advanced_algorithms/Quantum_Computing_Quantum_Monte_Carlo.ipynb) | [Motta2018](https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1364), [Peruzzo2014](https://www.nature.com/articles/ncomms5213) |


| Auxiliary functions | Notebook |
| ----- | ----- |
| Random circuit generator | [Random_Circuit.ipynb](notebooks/auxiliary_functions/Random_Circuit.ipynb) |

---
### Community repos

> :warning: **The following includes projects that are not provided by Amazon Braket. You are solely responsible for your use of those projects (including compliance with any applicable licenses and fitness of the project for your particular purpose).**

Quantum algorithm implementations using Braket in other repos:

| Algorithm | Repo | References | Additional dependencies |
| ----- | ----- | ----- | ----- |
| Quantum Reinforcement Learning | [quantum-computing-exploration-for-drug-discovery-on-aws](https://github.com/awslabs/quantum-computing-exploration-for-drug-discovery-on-aws)| [Learning Retrosynthetic Planning through Simulated Experience(2019)](https://pubs.acs.org/doi/10.1021/acscentsci.9b00055) | [dependencies](https://github.com/awslabs/quantum-computing-exploration-for-drug-discovery-on-aws/blob/main/source/src/notebook/healthcare-and-life-sciences/d-1-retrosynthetic-planning-quantum-reinforcement-learning/requirements.txt)

[comment]: <> (If you wish to highlight your implementation,  append the following content in a new line to the table above : | <Name> | <link to github repo> | <published reference> | <list of required packages on top of what is listed in amazon-braket-algorithm-library setup.py> |)

---
## <a name="install">Installing the Amazon Braket Algorithm Library</a>
The Amazon Braket Algorithm Library can be installed from source by cloning this repository and running a pip install command in the root directory of the repository.

```bash
git clone https://github.com/amazon-braket/amazon-braket-algorithm-library.git
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
the issue on our [GitHub issue tracker](https://github.com/amazon-braket/amazon-braket-algorithm-library/issues).  
For other issues or general questions, please ask on the [Quantum Computing Stack Exchange](https://quantumcomputing.stackexchange.com/questions/ask) and add the tag amazon-braket.

### Feedback and Feature Requests

If you have feedback or features that you would like to see on Amazon Braket, we would love to hear from you!  
[GitHub issues](https://github.com/amazon-braket/amazon-braket-algorithm-library/issues) is our preferred mechanism for collecting feedback and feature requests, allowing other users 
to engage in the conversation, and +1 issues to help drive priority. 


## License
This project is licensed under the Apache-2.0 License.
