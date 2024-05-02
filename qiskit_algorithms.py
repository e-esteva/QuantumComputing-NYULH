# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Amplification problem class."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, List, cast

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit.quantum_info import Statevector


class AmplificationProblem:
    """The amplification problem is the input to amplitude amplification algorithms, like Grover.

    This class contains all problem-specific information required to run an amplitude amplification
    algorithm. It minimally contains the Grover operator. It can further hold some post processing
    on the optimal bitstring.
    """

    def __init__(
        self,
        oracle: QuantumCircuit | Statevector,
        state_preparation: QuantumCircuit | None = None,
        grover_operator: QuantumCircuit | None = None,
        post_processing: Callable[[str], Any] | None = None,
        objective_qubits: int | list[int] | None = None,
        is_good_state: Callable[[str], bool] | list[int] | list[str] | Statevector | None = None,
    ) -> None:
        r"""
        Args:
            oracle: The oracle reflecting about the bad states.
            state_preparation: A circuit preparing the input state, referred to as
                :math:`\mathcal{A}`. If None, a layer of Hadamard gates is used.
            grover_operator: The Grover operator :math:`\mathcal{Q}` used as unitary in the
                phase estimation circuit. If None, this operator is constructed from the ``oracle``
                and ``state_preparation``.
            post_processing: A mapping applied to the most likely bitstring.
            objective_qubits: If set, specifies the indices of the qubits that should be measured.
                If None, all qubits will be measured. The ``is_good_state`` function will be
                applied on the measurement outcome of these qubits.
            is_good_state: A function to check whether a string represents a good state. By default
                if the ``oracle`` argument has an ``evaluate_bitstring`` method (currently only
                provided by the :class:`~qiskit.circuit.library.PhaseOracle` class) this will be
                used, otherwise this kwarg is required and **must** be specified.
        """
        self._oracle = oracle
        self._state_preparation = state_preparation
        self._grover_operator = grover_operator
        self._post_processing = post_processing
        self._objective_qubits = objective_qubits
        if is_good_state is not None:
            self._is_good_state = is_good_state
        elif hasattr(oracle, "evaluate_bitstring"):
            self._is_good_state = oracle.evaluate_bitstring
        else:
            self._is_good_state = None

    @property
    def oracle(self) -> QuantumCircuit | Statevector:
        """Return the oracle.

        Returns:
            The oracle.
        """
        return self._oracle

    @oracle.setter
    def oracle(self, oracle: QuantumCircuit | Statevector) -> None:
        """Set the oracle.

        Args:
            oracle: The oracle.
        """
        self._oracle = oracle

    @property
    def state_preparation(self) -> QuantumCircuit:
        r"""Get the state preparation operator :math:`\mathcal{A}`.

        Returns:
            The :math:`\mathcal{A}` operator as `QuantumCircuit`.
        """
        if self._state_preparation is None:
            state_preparation = QuantumCircuit(self.oracle.num_qubits)
            state_preparation.h(state_preparation.qubits)
            return state_preparation

        return self._state_preparation

    @state_preparation.setter
    def state_preparation(self, state_preparation: QuantumCircuit | None) -> None:
        r"""Set the :math:`\mathcal{A}` operator. If None, a layer of Hadamard gates is used.

        Args:
            state_preparation: The new :math:`\mathcal{A}` operator or None.
        """
        self._state_preparation = state_preparation

    @property
    def post_processing(self) -> Callable[[str], Any]:
        """Apply post processing to the input value.

        Returns:
            A handle to the post processing function. Acts as identity by default.
        """
        if self._post_processing is None:
            return lambda x: x

        return self._post_processing

    @post_processing.setter
    def post_processing(self, post_processing: Callable[[str], Any]) -> None:
        """Set the post processing function.

        Args:
            post_processing: A handle to the post processing function.
        """
        self._post_processing = post_processing

    @property
    def objective_qubits(self) -> list[int]:
        """The indices of the objective qubits.

        Returns:
            The indices of the objective qubits as list of integers.
        """
        if self._objective_qubits is None:
            return list(range(self.oracle.num_qubits))

        if isinstance(self._objective_qubits, int):
            return [self._objective_qubits]

        return self._objective_qubits

    @objective_qubits.setter
    def objective_qubits(self, objective_qubits: int | list[int] | None) -> None:
        """Set the objective qubits.

        Args:
            objective_qubits: The indices of the qubits that should be measured.
                If None, all qubits will be measured. The ``is_good_state`` function will be
                applied on the measurement outcome of these qubits.
        """
        self._objective_qubits = objective_qubits

    @property
    def is_good_state(self) -> Callable[[str], bool]:
        """Check whether a provided bitstring is a good state or not.

        Returns:
            A callable that takes in a bitstring and returns True if the measurement is a good
            state, False otherwise.
        """
        if (self._is_good_state is None) or callable(self._is_good_state):
            return self._is_good_state  # returns None if no is_good_state arg has been set
        elif isinstance(self._is_good_state, list):
            if all(isinstance(good_bitstr, str) for good_bitstr in self._is_good_state):
                return lambda bitstr: bitstr in cast(List[str], self._is_good_state)
            else:
                return lambda bitstr: all(
                    bitstr[good_index] == "1" for good_index in cast(List[int], self._is_good_state)
                )

        return lambda bitstr: bitstr in cast(Statevector, self._is_good_state).probabilities_dict()

    @is_good_state.setter
    def is_good_state(
        self, is_good_state: Callable[[str], bool] | list[int] | list[str] | Statevector
    ) -> None:
        """Set the ``is_good_state`` function.

        Args:
            is_good_state: A function to determine whether a bitstring represents a good state.
        """
        self._is_good_state = is_good_state

    @property
    def grover_operator(self) -> QuantumCircuit | None:
        r"""Get the :math:`\mathcal{Q}` operator, or Grover operator.

        If the Grover operator is not set, we try to build it from the :math:`\mathcal{A}` operator
        and `objective_qubits`. This only works if `objective_qubits` is a list of integers.

        Returns:
            The Grover operator, or None if neither the Grover operator nor the
            :math:`\mathcal{A}` operator is  set.
        """
        if self._grover_operator is None:
            return GroverOperator(self.oracle, self.state_preparation)
        return self._grover_operator

    @grover_operator.setter
    def grover_operator(self, grover_operator: QuantumCircuit | None) -> None:
        r"""Set the :math:`\mathcal{Q}` operator.

        If None, this operator is constructed from the ``oracle`` and ``state_preparation``.

        Args:
            grover_operator: The new :math:`\mathcal{Q}` operator or None.
        """
        self._grover_operator = grover_operator



# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
This module implements the abstract base class for algorithm results.
"""

from abc import ABC
import inspect
import pprint


class AlgorithmResult(ABC):
    """Abstract Base Class for algorithm results."""

    def __str__(self) -> str:
        result = {}
        for name, value in inspect.getmembers(self):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and not inspect.isfunction(value)
                and hasattr(self, name)
            ):

                result[name] = value

        return pprint.pformat(result, indent=4)

    def combine(self, result: "AlgorithmResult") -> None:
        """
        Any property from the argument that exists in the receiver is
        updated.
        Args:
            result: Argument result with properties to be set.
        Raises:
            TypeError: Argument is None
        """
        if result is None:
            raise TypeError("Argument result expected.")
        if result == self:
            return

        # find any result public property that exists in the receiver
        for name, value in inspect.getmembers(result):
            if (
                not name.startswith("_")
                and not inspect.ismethod(value)
                and not inspect.isfunction(value)
                and hasattr(self, name)
            ):
                try:
                    setattr(self, name, value)
                except AttributeError:
                    # some attributes may be read only
                    pass




# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The interface for amplification algorithms and results."""

from abc import ABC, abstractmethod
from typing import Any

#from .amplification_problem import AmplificationProblem
#from ..algorithm_result import AlgorithmResult


class AmplitudeAmplifier(ABC):
    """The interface for amplification algorithms."""

    @abstractmethod
    def amplify(self, amplification_problem: AmplificationProblem) -> "AmplitudeAmplifierResult":
        """Run the amplification algorithm.

        Args:
            amplification_problem: The amplification problem.

        Returns:
            The result as a ``AmplificationResult``, where e.g. the most likely state can be queried
            as ``result.top_measurement``.
        """
        raise NotImplementedError


class AmplitudeAmplifierResult(AlgorithmResult):
    """The amplification result base class."""

    def __init__(self) -> None:
        super().__init__()
        self._top_measurement: str | None = None
        self._assignment = None
        self._oracle_evaluation: bool | None = None
        self._circuit_results: list[dict[str, int]] | None = None
        self._max_probability: float | None = None

    @property
    def top_measurement(self) -> str | None:
        """The most frequently measured output as bitstring.

        Returns:
            The most frequently measured output state.
        """
        return self._top_measurement

    @top_measurement.setter
    def top_measurement(self, value: str) -> None:
        """Set the most frequently measured bitstring.

        Args:
            value: A new value for the top measurement.
        """
        self._top_measurement = value

    @property
    def assignment(self) -> Any:
        """The post-processed value of the most likely bitstring.

        Returns:
            The output of the ``post_processing`` function of the respective
            ``AmplificationProblem``, where the input is the ``top_measurement``. The type
            is the same as the return type of the post-processing function.
        """
        return self._assignment

    @assignment.setter
    def assignment(self, value: Any) -> None:
        """Set the value for the assignment.

        Args:
            value: A new value for the assignment/solution.
        """
        self._assignment = value

    @property
    def oracle_evaluation(self) -> bool:
        """Whether the classical oracle evaluation of the top measurement was True or False.

        Returns:
            The classical oracle evaluation of the top measurement.
        """
        return self._oracle_evaluation

    @oracle_evaluation.setter
    def oracle_evaluation(self, value: bool) -> None:
        """Set the classical oracle evaluation of the top measurement.

        Args:
            value: A new value for the classical oracle evaluation.
        """
        self._oracle_evaluation = value

    @property
    def circuit_results(self) -> list[dict[str, int]] | None:
        """Return the circuit results."""
        return self._circuit_results

    @circuit_results.setter
    def circuit_results(self, value: list[dict[str, int]]) -> None:
        """Set the circuit results."""
        self._circuit_results = value

    @property
    def max_probability(self) -> float:
        """Return the maximum sampling probability."""
        return self._max_probability

    @max_probability.setter
    def max_probability(self, value: float) -> None:
        """Set the maximum sampling probability."""
        self._max_probability = value













# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Grover's search algorithm."""

import itertools
from collections.abc import Iterator, Generator
from typing import Any

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.quantum_info import Statevector

#from qiskit_algorithms.exceptions import AlgorithmError
#from qiskit_algorithms.utils import algorithm_globals

#from .amplification_problem import AmplificationProblem
#from .amplitude_amplifier import AmplitudeAmplifier, AmplitudeAmplifierResult

class Grover(AmplitudeAmplifier):
    r"""Grover's Search algorithm.

    .. note::

        If you want to learn more about the theory behind Grover's Search algorithm, check
        out the `Qiskit Textbook <https://qiskit.org/textbook/ch-algorithms/grover.html>`_.
        or the `Qiskit Tutorials
        <https://qiskit.org/documentation/tutorials/algorithms/07_grover_examples.html>`_
        for more concrete how-to examples.

    Grover's Search [1, 2] is a well known quantum algorithm that can be used for
    searching through unstructured collections of records for particular targets
    with quadratic speedup compared to classical algorithms.

    Given a set :math:`X` of :math:`N` elements :math:`X=\{x_1,x_2,\ldots,x_N\}`
    and a boolean function :math:`f : X \rightarrow \{0,1\}`, the goal of an
    unstructured-search problem is to find an element :math:`x^* \in X` such
    that :math:`f(x^*)=1`.

    The search is called *unstructured* because there are no guarantees as to how
    the database is ordered.  On a sorted database, for instance, one could perform
    binary search to find an element in :math:`\mathbb{O}(\log N)` worst-case time.
    Instead, in an unstructured-search problem, there is no prior knowledge about
    the contents of the database. With classical circuits, there is no alternative
    but to perform a linear number of queries to find the target element.
    Conversely, Grover's Search algorithm allows to solve the unstructured-search
    problem on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries.

    To carry out this search a so-called oracle is required, that flags a good element/state.
    The action of the oracle :math:`\mathcal{S}_f` is

    .. math::

        \mathcal{S}_f |x\rangle = (-1)^{f(x)} |x\rangle,

    i.e. it flips the phase of the state :math:`|x\rangle` if :math:`x` is a hit.
    The details of how :math:`S_f` works are unimportant to the algorithm; Grover's
    search algorithm treats the oracle as a black box.

    This class supports oracles in form of a :class:`~qiskit.circuit.QuantumCircuit`.

    With the given oracle, Grover's Search constructs the Grover operator to amplify the
    amplitudes of the good states:

    .. math::

        \mathcal{Q} = H^{\otimes n} \mathcal{S}_0 H^{\otimes n} \mathcal{S}_f
                    = D \mathcal{S}_f,

    where :math:`\mathcal{S}_0` flips the phase of the all-zero state and acts as identity
    on all other states. Sometimes the first three operands are summarized as diffusion operator,
    which implements a reflection over the equal superposition state.

    If the number of solutions is known, we can calculate how often :math:`\mathcal{Q}` should be
    applied to find a solution with very high probability, see the method
    `optimal_num_iterations`. If the number of solutions is unknown, the algorithm tries different
    powers of Grover's operator, see the `iterations` argument, and after each iteration checks
    if a good state has been measured using `good_state`.

    The generalization of Grover's Search, Quantum Amplitude Amplification [3], uses a modified
    version of :math:`\mathcal{Q}` where the diffusion operator does not reflect about the
    equal superposition state, but another state specified via an operator :math:`\mathcal{A}`:

    .. math::

        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f.

    For more information, see the :class:`~qiskit.circuit.library.GroverOperator` in the
    circuit library.

    References:
        [1]: L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
            `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.
        [2]: I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
            Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.
        [3]: Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
            Quantum Amplitude Amplification and Estimation.
            `arXiv:quant-ph/0005055 <http://arxiv.org/abs/quant-ph/0005055>`_.
    """

    def __init__(
        self,
        iterations: list[int] | Iterator[int] | int | None = None,
        growth_rate: float | None = None,
        sample_from_iterations: bool = False,
        sampler: BaseSampler | None = None,
    ) -> None:
        r"""
        Args:
            iterations: Specify the number of iterations/power of Grover's operator to be checked.
                * If an int, only one circuit is run with that power of the Grover operator.
                If the number of solutions is known, this option should be used with the optimal
                power. The optimal power can be computed with ``Grover.optimal_num_iterations``.
                * If a list, all the powers in the list are run in the specified order.
                * If an iterator, the powers yielded by the iterator are checked, until a maximum
                number of iterations or maximum power is reached.
                * If ``None``, the :obj:`AmplificationProblem` provided must have an ``is_good_state``,
                and circuits are run until that good state is reached.
            growth_rate: If specified, the iterator is set to increasing powers of ``growth_rate``,
                i.e. to ``int(growth_rate ** 1), int(growth_rate ** 2), ...`` until a maximum
                number of iterations is reached.
            sample_from_iterations: If True, instead of taking the values in ``iterations`` as
                powers of the Grover operator, a random integer sample between 0 and smaller value
                than the iteration is used as a power, see [1], Section 4.
            sampler: A Sampler to use for sampling the results of the circuits.

        Raises:
            ValueError: If ``growth_rate`` is a float but not larger than 1.
            ValueError: If both ``iterations`` and ``growth_rate`` is set.

        References:
            [1]: Boyer et al., Tight bounds on quantum searching
                 `<https://arxiv.org/abs/quant-ph/9605034>`_
        """
        # set default value
        if growth_rate is None and iterations is None:
            growth_rate = 1.2

        if growth_rate is not None and iterations is not None:
            raise ValueError("Pass either a value for iterations or growth_rate, not both.")

        if growth_rate is not None:
            # yield iterations ** 1, iterations ** 2, etc. and casts to int
            self._iterations: Generator[int, None, None] | list[int] = (
                int(growth_rate**x) for x in itertools.count(1)
            )
        elif isinstance(iterations, int):
            self._iterations = [iterations]
        else:
            self._iterations = iterations  # type: ignore[assignment]

        self._sampler = sampler
        self._sample_from_iterations = sample_from_iterations
        self._iterations_arg = iterations

    @property
    def sampler(self) -> BaseSampler | None:
        """Get the sampler.

        Returns:
            The sampler used to run this algorithm.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: BaseSampler) -> None:
        """Set the sampler.

        Args:
            sampler: The sampler used to run this algorithm.
        """
        self._sampler = sampler

    def amplify(self, amplification_problem: AmplificationProblem) -> "GroverResult":
        """Run the Grover algorithm.

        Args:
            amplification_problem: The amplification problem.

        Returns:
            The result as a ``GroverResult``, where e.g. the most likely state can be queried
            as ``result.top_measurement``.

        Raises:
            ValueError: If sampler is not set.
            AlgorithmError: If sampler job fails.
            TypeError: If ``is_good_state`` is not provided and is required (i.e. when iterations
            is ``None`` or a ``list``)
        """
        if self._sampler is None:
            raise ValueError("A sampler must be provided.")

        if isinstance(self._iterations, list):
            max_iterations = len(self._iterations)
            max_power = np.inf  # no cap on the power
            iterator: Iterator[int] = iter(self._iterations)
        else:
            max_iterations = max(10, 2**amplification_problem.oracle.num_qubits)
            max_power = np.ceil(
                2 ** (len(amplification_problem.grover_operator.reflection_qubits) / 2)
            )
            iterator = self._iterations

        result = GroverResult()

        iterations = []
        top_measurement = "0" * len(amplification_problem.objective_qubits)
        oracle_evaluation = False
        all_circuit_results = []
        max_probability = 0

        for _ in range(max_iterations):  # iterate at most to the max number of iterations
            # get next power and check if allowed
            power = next(iterator)

            if power > max_power:
                break

            iterations.append(power)  # store power

            # sample from [0, power) if specified
            if self._sample_from_iterations:
                power = algorithm_globals.random.integers(power)
            # Run a grover experiment for a given power of the Grover operator.
            if self._sampler is not None:
                qc = self.construct_circuit(amplification_problem, power, measurement=True)
                job = self._sampler.run([qc])

                try:
                    results = job.result()
                except Exception as exc:
                    raise AlgorithmError("Sampler job failed.") from exc

                num_bits = len(amplification_problem.objective_qubits)
                circuit_results: dict[str, Any] | Statevector | np.ndarray = {
                    np.binary_repr(k, num_bits): v for k, v in results.quasi_dists[0].items()
                }
                top_measurement, max_probability = max(
                    circuit_results.items(), key=lambda x: x[1]  # type: ignore[union-attr]
                )

            all_circuit_results.append(circuit_results)

            if (isinstance(self._iterations_arg, int)) and (
                amplification_problem.is_good_state is None
            ):
                oracle_evaluation = None  # cannot check for good state without is_good_state arg
                break

            # is_good_state arg must be provided if iterations arg is not an integer
            if (
                self._iterations_arg is None or isinstance(self._iterations_arg, list)
            ) and amplification_problem.is_good_state is None:
                raise TypeError("An is_good_state function is required with the provided oracle")

            # only check if top measurement is a good state if an is_good_state arg is provided
            oracle_evaluation = amplification_problem.is_good_state(top_measurement)

            if oracle_evaluation is True:
                break  # we found a solution

        result.iterations = iterations
        result.top_measurement = top_measurement
        result.assignment = amplification_problem.post_processing(top_measurement)
        result.oracle_evaluation = oracle_evaluation
        result.circuit_results = all_circuit_results  # type: ignore[assignment]
        result.max_probability = max_probability

        return result

    @staticmethod
    def optimal_num_iterations(num_solutions: int, num_qubits: int) -> int:
        """Return the optimal number of iterations, if the number of solutions is known.

        Args:
            num_solutions: The number of solutions.
            num_qubits: The number of qubits used to encode the states.

        Returns:
            The optimal number of iterations for Grover's algorithm to succeed.
        """
        amplitude = np.sqrt(num_solutions / 2**num_qubits)
        return round(np.arccos(amplitude) / (2 * np.arcsin(amplitude)))

    def construct_circuit(
        self, problem: AmplificationProblem, power: int | None = None, measurement: bool = False
    ) -> QuantumCircuit:
        """Construct the circuit for Grover's algorithm with ``power`` Grover operators.

        Args:
            problem: The amplification problem for the algorithm.
            power: The number of times the Grover operator is repeated. If None, this argument
                is set to the first item in ``iterations``.
            measurement: Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit

        Raises:
            ValueError: If no power is passed and the iterations are not an integer.
        """
        if power is None:
            if len(self._iterations) > 1:  # type: ignore[arg-type]
                raise ValueError("Please pass ``power`` if the iterations are not an integer.")
            power = self._iterations[0]  # type: ignore[index]

        qc = QuantumCircuit(problem.oracle.num_qubits, name="Grover circuit")
        qc.compose(problem.state_preparation, inplace=True)
        if power > 0:
            qc.compose(problem.grover_operator.power(power), inplace=True)

        if measurement:
            measurement_cr = ClassicalRegister(len(problem.objective_qubits))
            qc.add_register(measurement_cr)
            qc.measure(problem.objective_qubits, measurement_cr)

        return qc


class GroverResult(AmplitudeAmplifierResult):
    """Grover Result."""

    def __init__(self) -> None:
        super().__init__()
        self._iterations: list[int] | None = None

    @property
    def iterations(self) -> list[int]:
        """All the powers of the Grover operator that have been tried.

        Returns:
            The powers of the Grover operator tested.
        """
        return self._iterations

    @iterations.setter
    def iterations(self, value: list[int]) -> None:
        """Set the powers of the Grover operator that have been tried.

        Args:
            value: A new value for the powers.
        """
        self._iterations = value
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception and warnings for errors raised by Algorithms module."""

from qiskit.exceptions import QiskitError


class AlgorithmError(QiskitError):
    """For Algorithm specific errors."""

    pass


class QiskitAlgorithmsWarning(UserWarning):
    """Base class for warnings raised by Qiskit Algorithms."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(" ".join(message))
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class QiskitAlgorithmsOptimizersWarning(QiskitAlgorithmsWarning):
    """For Algorithm specific warnings."""

    pass


# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
utils.algorithm_globals
=======================
Common (global) properties used across qiskit_algorithms.

.. currentmodule:: qiskit_algorithms.utils.algorithm_globals

Includes:

  * Random number generator and random seed.

    Algorithms can use the generator for random values, as needed, and it
    can be seeded here for reproducible results when using such an algorithm.
    This is often important, for example in unit tests, where the same
    outcome is desired each time (reproducible) and not have it be variable
    due to randomness.

Attributes:
    random_seed (int | None): Random generator seed (read/write).
    random (np.random.Generator): Random generator (read-only)
"""

import warnings

import numpy as np


class QiskitAlgorithmGlobals:
    """Global properties for algorithms."""

    # The code is done to work even after some future removal of algorithm_globals
    # from Qiskit (qiskit.utils). All that is needed in the future, after that, if
    # this is updated, is just the logic in the except blocks.
    #
    # If the Qiskit version exists this acts a redirect to that (it delegates the
    # calls off to it). In the future when that does not exist this has similar code
    # in the except blocks here, as noted above, that will take over. By delegating
    # to the Qiskit instance it means that any existing code that uses that continues
    # to work. Logic here in qiskit_algorithms though uses this instance and the
    # random check here has logic to warn if the seed here is not the same as the Qiskit
    # version so we can detect direct usage of the Qiskit version and alert the user to
    # change their code to use this. So simply changing from:
    #     from qiskit.utils import algorithm_globals
    # to
    #     from qiskit_algorithm.utils import algorithm_globals

    def __init__(self) -> None:
        self._random_seed: int | None = None
        self._random: np.random.Generator | None = None

    @property
    def random_seed(self) -> int | None:
        """Random seed property (getter/setter)."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                return qiskit_globals.random_seed

        except ImportError:
            return self._random_seed

    @random_seed.setter
    def random_seed(self, seed: int | None) -> None:
        """Set the random generator seed.

        Args:
            seed: If ``None`` then internally a random value is used as a seed
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                qiskit_globals.random_seed = seed
                # Mirror the seed here when set via this random_seed. If the seed is
                # set on the qiskit.utils instance then we can detect it's different
                self._random_seed = seed

        except ImportError:
            self._random_seed = seed
            self._random = None

    @property
    def random(self) -> np.random.Generator:
        """Return a numpy np.random.Generator (default_rng) using random_seed."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)

                from qiskit.utils import algorithm_globals as qiskit_globals

                if self._random_seed != qiskit_globals.random_seed:
                    # If the seeds are different - likely this local is None and the qiskit.utils
                    # algorithms global was seeded directly then we will warn to use this here as
                    # the Qiskit version is planned to be removed in a future version of Qiskit.
                    warnings.warn(
                        "Using random that is seeded via qiskit.utils algorithm_globals is deprecated "
                        "since version 0.2.0. Instead set random_seed directly to "
                        "qiskit_algorithms.utils algorithm_globals.",
                        category=DeprecationWarning,
                        stacklevel=2,
                    )

                return qiskit_globals.random

        except ImportError:
            if self._random is None:
                self._random = np.random.default_rng(self._random_seed)
            return self._random


# Global instance to be used as the entry point for globals.
algorithm_globals = QiskitAlgorithmGlobals()
