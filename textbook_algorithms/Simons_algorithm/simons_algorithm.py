from braket.circuits import Circuit, circuit
from braket.aws import AwsDevice
from sympy import Matrix
import numpy as np


@circuit.subroutine(register=True)
def simons_oracle(secret_s: str):
    """
    Quantum circuit implementing a particular oracle for Simon's problem. Details of this implementation are
    explained in the Simons Algorithm demo notebook.
    Args:
        secret_s (str): secret string we wish to find
    """
    # Find the index of the first 1 in s, to be used as the flag bit
    flag_bit=secret_s.find('1')
    
    n=len(secret_s)
    
    circ = Circuit()
    # First copy the first n qubits, so that |x>|0> -> |x>|x>
    for i in range(n):
        circ.cnot(i, i+n)
    
    # If flag_bit=-1, s is the all-zeros string, and we do nothing else.
    if flag_bit != -1:
        # Now apply the XOR with s whenever the flag bit is 1.
        for index,bit_value in enumerate(secret_s):
            
            if bit_value not in ['0','1']:
                raise Exception ('Incorrect char \'' + bit_value + '\' in secret string s:' + secret_s)
                
            # XOR with s whenever the flag bit is 1.
            # In terms of gates, XOR means we apply an X gate only whenever the corresponding bit in s is 1.
            # Applying this X only when the flag qubit is 1 means this is a CNOT gate.
            if(bit_value == '1'):
                circ.cnot(flag_bit,index+n)
    return circ

def simons_circuit(secret_s: str):
    n = len(secret_s)
    return Circuit().h(range(n)).simons_oracle(secret_s).h(range(n)) 

def submit_simons_tasks(secret_s: str, device: AwsDevice):
    return device.run(simons_circuit(secret_s), shots=4*len(secret_s))

def process_simons_results(task):
    result = task.result()

    n = int(len(result.measured_qubits)/2)

    new_results = {}
    for bitstring, count in result.measurement_counts.items():
        # Only keep the outcomes on first n qubits
        trunc_bitstring = bitstring[:n]
        # Add the count to that of the of truncated bit string
        new_results[trunc_bitstring] = new_results.get(trunc_bitstring, 0) + count

    if len(new_results.keys()) < n:
        raise Exception ('System will be underdetermined. Minimum ' + str(n) + ' bistrings needed, but only '
                        + str(len(new_results.keys())) +' returned. Please rerun Simon\'s algorithm.')
    string_list = []

    for key in new_results.keys():
    #     if key!= "0"*n:
        string_list.append( [ int(c) for c in key ] )

    M=Matrix(string_list).T

    # Construct the agumented matrix
    M_I = Matrix(np.hstack([M,np.eye(M.shape[0],dtype=int)]))

    # Perform row reduction, working modulo 2. We use the iszerofunc property of rref
    # to perform the Gaussian elimination over the finite field.
    M_I_rref = M_I.rref(iszerofunc=lambda x: x % 2==0)

    # In row reduced echelon form, we can end up with a solution outside of the finite field {0,1}.
    # Thus, we need to revert the matrix back to this field by treating fractions as a modular inverse.
    # Since the denominator will always be odd (i.e. 1 mod 2), it can be ignored.

    # Helper function to treat fractions as modular inverse:
    def mod2(x):
        return x.as_numer_denom()[0] % 2

    # Apply our helper function to the matrix
    M_I_final = M_I_rref[0].applyfunc(mod2)

    # Extract the kernel of M from the remaining columns of the last row, when s is nonzero.
    if all(value == 0 for value in M_I_final[-1,:M.shape[1]]):
        result_s="".join(str(c) for c in M_I_final[-1,M.shape[1]:])

    # Otherwise, the sub-matrix will be full rank, so just set s=0...0
    else:
        result_s='0'*M.shape[0]

    return result_s
