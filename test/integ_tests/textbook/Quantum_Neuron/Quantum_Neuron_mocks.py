import tarfile


def pre_run_inject(mock_utils):
    mocker = mock_utils.Mocker()
    mock_utils.mock_default_device_calls(mocker)
    mocker.set_search_result([
        {
            "Roles" : [
                {
                    "RoleName": "AmazonBraketJobsExecutionRole",
                    "Arn" : "TestRoleARN"
                }
            ]
        }
    ])
    mocker.set_create_job_result({
        "jobArn" : f"arn:aws:braket:{mocker.region_name}:000000:job/testJob"
    })
    mocker.set_get_job_result({
        "instanceConfig" : {
            "instanceCount" : 1
        },
        "jobName": "testJob",
        "status": "COMPLETED",
        "outputDataConfig": {
            "s3Path" : "s3://amazon-br-invalid-path/test-path/test-results"
        },
        "checkpointConfig": {
            "s3Uri" : "s3://amazon-br-invalid-path/test-path/test-results"
        }
    })
    mocker.set_log_streams_result({
        "logStreams": []
    })
    mocker.set_task_result_return(mock_utils.read_file("quantum_neuron_mock_results.json", __file__))
    default_job_results = mock_utils.read_file("quantum_neuron_mock_results.json", __file__)
    with open("results.json", "w") as f:
        f.write(default_job_results)
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("results.json")


def post_run(tb):
    pass
