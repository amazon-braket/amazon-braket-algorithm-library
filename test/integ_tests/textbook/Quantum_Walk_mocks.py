def pre_run_inject_2(mock_utils):
    pass


def pre_run_inject(mock_utils):
    mocker = mock_utils.Mocker()
    mock_utils.mock_default_device_calls(mocker)
    res1 = mock_utils.read_file("0_quantum_walk.json", __file__)
    res2 = mock_utils.read_file("1_quantum_walk.json", __file__)
    res3 = mock_utils.read_file("2_quantum_walk.json", __file__)
    res4 = mock_utils.read_file("3_quantum_walk.json", __file__)
    effects = []
    for i in range(3):
        effects.append(res1)
    for i in range(51):
        effects.append(res2)
    for i in range(20):
        effects.append(res3)
    for i in range(20):
        effects.append(res4)
    mocker.set_task_result_side_effect(effects)


def post_run(tb):
    pass
