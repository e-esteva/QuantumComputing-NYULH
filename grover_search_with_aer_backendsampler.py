
from qiskit_algorithms import Grover
from qiskit.primitives import Sampler
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator

print("RUNNING IDEAL SIM ON AER")

# instantiating gpu-powered simulator with 'automatic' settings
simulator = AerSimulator()
# defining the aer sim as the backend for the backend sampler
sampler = BackendSampler(simulator)
# confirming we're using GPUs
simulator.available_devices()


# run grover search using aer-backed sampler:
grover = Grover(sampler=sampler)


from qiskit import QuantumCircuit
from qiskit_algorithms import AmplificationProblem

# the state we desire to find is '11'
good_state = ["11"]

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# define Grover's algorithm
problem = AmplificationProblem(oracle, is_good_state=good_state)

# now we can have a look at the Grover operator that is used in running the algorithm
# (Algorithm circuits are wrapped in a gate to appear in composition as a block
# so we have to decompose() the op to see it expanded into its component gates.)
#problem.grover_operator.decompose().draw(output="mpl")


result = grover.amplify(problem)
print("Result type:", type(result))
print()
print("Success!" if result.oracle_evaluation else "Failure!")
print("Top measurement:", result.top_measurement)
print('Result: '+str(result.circuit_results))

############################################################## Manila fake backend (error model from real quantum device):
print('RUNNING FAKE MANILA V2 BACKEND')

from qiskit_ibm_runtime.fake_provider import FakeManilaV2
backend = FakeManilaV2()
sampler = BackendSampler(backend)


# run grover search using aer-backed sampler:
grover = Grover(sampler=sampler)


from qiskit import QuantumCircuit
from qiskit_algorithms import AmplificationProblem

# the state we desire to find is '11'
good_state = ["11"]

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# define Grover's algorithm
problem = AmplificationProblem(oracle, is_good_state=good_state)


result = grover.amplify(problem)
print("Result type:", type(result))
print()
print("Success!" if result.oracle_evaluation else "Failure!")
print("Top measurement:", result.top_measurement)
print('Result: '+str(result.circuit_results))

################################################################ now running Manila backend on aer simulator with GPUs:
# backend is same as above (Manila2)
print('RUNNING FAKE MANILA V2 BACKEND ON AER GPU SIM')

sim_manila = AerSimulator.from_backend(backend)
sampler = BackendSampler(sim_manila)
grover = Grover(sampler=sampler)
result = grover.amplify(problem)




# confirming we're using GPUs
print(sim_manila.available_devices())


# run grover search using aer-backed sampler:
grover = Grover(sampler=sampler)


from qiskit import QuantumCircuit
from qiskit_algorithms import AmplificationProblem

# the state we desire to find is '11'
good_state = ["11"]

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)

# define Grover's algorithm
problem = AmplificationProblem(oracle, is_good_state=good_state)

# now we can have a look at the Grover operator that is used in running the algorithm
# (Algorithm circuits are wrapped in a gate to appear in composition as a block
# so we have to decompose() the op to see it expanded into its component gates.)
#problem.grover_operator.decompose().draw(output="mpl")


result = grover.amplify(problem)
print("Result type:", type(result))
print()
print("Success!" if result.oracle_evaluation else "Failure!")
print("Top measurement:", result.top_measurement)
print('Result: ' +str(result.circuit_results))
