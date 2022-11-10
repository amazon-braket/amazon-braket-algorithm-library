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
from braket.circuits import Circuit


def example_function(my_param: int) -> Circuit:
    """This is an example function.

    Args:
        my_param (int): This is a parameter.

    Returns:
        Circuit: The circuit.
    """
    print(f"This is a test with {my_param}")
    circ = Circuit().h(my_param).cnot(0, 1)
    return circ
