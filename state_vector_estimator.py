from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Pauli, SparsePauliOp
 
import matplotlib.pyplot as plt
import numpy as np
 
# Define a circuit with two parameters.
circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)
circuit.ry(Parameter("a"), 0)
circuit.rz(Parameter("b"), 0)
circuit.cx(0, 1)
circuit.h(0)
 
# Define a sweep over parameter values, where the second axis is over
# the two parameters in the circuit.
params = np.vstack([
    np.linspace(-np.pi, np.pi, 100),
    np.linspace(-4 * np.pi, 4 * np.pi, 100)
]).T
 
# Define three observables. Many formats are supported here including
# classes such as qiskit.quantum_info.SparsePauliOp. The inner length-1
# lists cause this array of observables to have shape (3, 1), rather
# than shape (3,) if they were omitted.
observables = [
    [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
    [Pauli("XX")],
    [Pauli("IY")]
]
 
# Instantiate a new statevector simulation based estimator object.
estimator = StatevectorEstimator()
 
# Estimate the expectation value for all 300 combinations of
# observables and parameter values, where the pub result will have
# shape (3, 100). This shape is due to our array of parameter
# bindings having shape (100,), combined with our array of observables
# having shape (3, 1)
pub = (circuit, observables, params)
job = estimator.run([pub])
 
# Extract the result for the 0th pub (this example only has one pub).
result = job.result()[0]
 
# Error-bar information is also available, but the error is 0
# for this StatevectorEstimator.
result.data.stds
 
# Pull out the array-based expectation value estimate data from the
# result and plot a trace for each observable.
for idx, pauli in enumerate(observables):
    plt.plot(result.data.evs[idx], label=pauli)
plt.legend()
