# Amazon Braket Algorithm Library
The Braket Algorithm Library provides Amazon Braket customers with 20 pre-built implementations of prominent quantum algorithms and experimental workloads as ready-to-run example notebooks.

---  

  * [**Grover's search algorithm**](samples/Grover-search-algorithm/Grover.ipynb)
  
    Grover's algorithm is arguably one of the canonical quantum algorithms that kick-started the field of quantum computing. In the future, it could possibly serve as a hallmark application of quantum computing. Grover's algorithm allows us to find a particular register in an unordered database with $N$ entries in just $O(\\sqrt{N})$.
  
  * [**Violation of Bell's Inequality**](samples/Grover-search-algorithm/Grover.ipynb)

    Bell's experiment considers that would happen if Alice and Bob each measure in two different directions.Alice will measure in $\mathbf{\hat a}$ and $\mathbf{\hat b}$, while Bob will measure in $\mathbf{\hat a}$ and $\mathbf{\hat c}$. The violation of Bell's Inequality rule out the hidden variable theory for quantum phenomena.

  * [**Bernstein–Vazirani algorithm**](samples/Grover-search-algorithm/Grover.ipynb)
  
    The Bernstein-Vazirani algorithm finds the hidden string with just a single application
    of the oracle. The algorithm was one of the first examples of a quasi-polynomial speed-up over a probabilistic classical computer. 
  
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


## License
This project is licensed under the Apache-2.0 License.
