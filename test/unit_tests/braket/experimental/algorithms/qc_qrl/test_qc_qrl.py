from unittest.mock import patch

import pytest

from braket.experimental.algorithms.qc_qrl.utility.RetroRLAgent import RetroRLAgent
from braket.experimental.algorithms.qc_qrl.utility.RetroGateModel import RetroGateModel

np.set_printoptions(precision=4, edgeitems=10, linewidth=150, suppress=True)


@pytest.fixture
def common_agent_param():
    agent_param = {}
    # initial the RetroRLModel object
    init_param = {}
    method = ["retro-rl", "retro-qrl"]

    for mt in method:
        if mt == "retro-rl":
            init_param[mt] = {}
            init_param[mt]["param"] = ["inputsize", "middlesize", "outputsize"]
        elif mt == "retro-qrl":
            init_param[mt] = {}
            init_param[mt]["param"] = ["n_qubits", "device", "framework", "shots", "layers"]

    agent_param["init_param"] = init_param
    # train_mode can be: "local-instance", "local-job", "hybrid-job"
    train_mode = "hybrid-job"

    data_path = "data"
    s3_data_path = None

    agent_param["data_path"] = data_path
    agent_param["s3_data_path"] = s3_data_path
    agent_param["train_mode"] = train_mode
    agent_param["episodes"] = 2

    return agent_param


def test_quantum_circuit_parameters(common_agent_param):
    agent_param = common_agent_param

    model_param = {}
    method = "retro-qrl"
    model_param[method] = {}
    model_param[method]["n_qubits"] = [8]
    model_param[method]["device"] = ["local"]
    model_param[method]["framework"] = ["pennylane"]
    model_param[method]["shots"] = [100]
    model_param[method]["layers"] = [1]

    agent_param["model_param"] = model_param

    n_qubits = model_param[method]["n_qubits"][0]
    device = model_param[method]["device"][0]
    framework = model_param[method]["framework"][0]
    shots = model_param[method]["shots"][0]
    layers = model_param[method]["layers"][0]

    model_name = "{}_{}_{}_{}_{}".format(n_qubits, device, framework, shots, layers)
    agent_param["model_name"] = model_name

    agent_param["train_mode"] = "local-instance"

    retro_qrl_agent = RetroRLAgent(build_model=True, method=method, **agent_param)

    quantum_param_sum = 0
    for param in retro_qrl_agent.NN.parameters():
        quantum_param_sum = quantum_param_sum + param.numel()

    assert quantum_param_sum == model_param[method]["n_qubits"][0]


def test_classical_circuit_parameters(common_agent_param):
    agent_param = common_agent_param

    model_param = {}
    method = "retro-rl"
    model_param[method] = {}
    model_param[method]["inputsize"] = [256]
    model_param[method]["middlesize"] = [256]
    model_param[method]["outputsize"] = [1]

    agent_param["model_param"] = model_param
    model_name = f"{model_param[method]['inputsize'][0]}_{model_param[method]['middlesize'][0]}_{model_param[method]['outputsize'][0]}"
    agent_param["model_name"] = model_name

    agent_param["train_mode"] = "local-instance"

    retro_crl_agent = RetroRLAgent(build_model=True, method=method, **agent_param)

    classical_param_sum = 0
    for param in retro_crl_agent.NN.parameters():
        classical_param_sum = classical_param_sum + param.numel()

    assert (
        classical_param_sum
        == model_param[method]["inputsize"][0]
        * model_param[method]["middlesize"][0]
        * model_param[method]["outputsize"][0]
    )
