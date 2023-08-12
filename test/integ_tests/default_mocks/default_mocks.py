def pre_run_inject(mock_utils):
    print("This is inside default_mocks.py")

    mocker = mock_utils.Mocker()
    mock_utils.mock_default_device_calls(mocker)
    mocker.set_create_job_result({"jobArn": "arn:aws:braket:us-east-1:000000:job/TestARN"})


def post_run(tb):
    pass
