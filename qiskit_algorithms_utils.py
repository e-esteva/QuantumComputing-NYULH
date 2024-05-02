from __future__ import annotations
# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common qiskit_algorithms utility functions."""

#from algorithm_globals import algorithm_globals
#from validate_initial_point import validate_initial_point
#from validate_bounds import validate_bounds

#__all__ = [
#    "algorithm_globals",
#    "validate_initial_point",
#    "validate_bounds",
#]



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


# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Additional optional constants.
"""

from qiskit.utils import LazyImportTester


HAS_NLOPT = LazyImportTester("nlopt", name="NLopt Optimizer", install="pip install nlopt")
HAS_SKQUANT = LazyImportTester(
    "skquant.opt",
    name="scikit-quant",
    install="pip install scikit-quant",
)
HAS_SQSNOBFIT = LazyImportTester("SQSnobFit", install="pip install SQSnobFit")
HAS_TWEEDLEDUM = LazyImportTester("tweedledum", install="pip install tweedledum")



# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Validate parameter bounds."""


from qiskit.circuit import QuantumCircuit


def validate_bounds(circuit: QuantumCircuit) -> list[tuple[float | None, float | None]]:
    """
    Validate the bounds provided by a quantum circuit against its number of parameters.
    If no bounds are obtained, return ``None`` for all lower and upper bounds.

    Args:
        circuit: A parameterized quantum circuit.

    Returns:
        A list of tuples (lower_bound, upper_bound)).

    Raises:
        ValueError: If the number of bounds does not the match the number of circuit parameters.
    """
    if hasattr(circuit, "parameter_bounds") and circuit.parameter_bounds is not None:
        bounds = circuit.parameter_bounds
        if len(bounds) != circuit.num_parameters:
            raise ValueError(
                f"The number of bounds ({len(bounds)}) does not match the number of "
                f"parameters in the circuit ({circuit.num_parameters})."
            )
    else:
        bounds = [(None, None)] * circuit.num_parameters

    return bounds


# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Validate an initial point."""


import numpy as np

from qiskit.circuit import QuantumCircuit
#from qiskit_algorithms.utils.algorithm_globals import algorithm_globals


def validate_initial_point(point: np.ndarray | None | None, circuit: QuantumCircuit) -> np.ndarray:
    r"""
    Validate a choice of initial point against a choice of circuit. If no point is provided, a
    random point will be generated within certain parameter bounds. It will first look to the
    circuit for these bounds. If the circuit does not specify bounds, bounds of :math:`-2\pi`,
    :math:`2\pi` will be used.

    Args:
        point: An initial point.
        circuit: A parameterized quantum circuit.

    Returns:
        A validated initial point.

    Raises:
        ValueError: If the dimension of the initial point does not match the number of circuit
        parameters.
    """
    expected_size = circuit.num_parameters

    if point is None:
        # get bounds if circuit has them set, otherwise use [-2pi, 2pi] for each parameter
        bounds = getattr(circuit, "parameter_bounds", None)
        if bounds is None:
            bounds = [(-2 * np.pi, 2 * np.pi)] * expected_size

        # replace all Nones by [-2pi, 2pi]
        lower_bounds = []
        upper_bounds = []
        for lower, upper in bounds:
            lower_bounds.append(lower if lower is not None else -2 * np.pi)
            upper_bounds.append(upper if upper is not None else 2 * np.pi)

        # sample from within bounds
        point = algorithm_globals.random.uniform(lower_bounds, upper_bounds)

    elif len(point) != expected_size:
        raise ValueError(
            f"The dimension of the initial point ({len(point)}) does not match the "
            f"number of parameters in the circuit ({expected_size})."
        )

    return point

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
Validation module
"""

from typing import Set


def validate_in_set(name: str, value: object, values: Set[object]) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        values: set that should contain value.
    Raises:
        ValueError: invalid value
    """
    if value not in values:
        raise ValueError(f"{name} must be one of '{values}', was '{value}'.")


def validate_min(name: str, value: float, minimum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum:
        raise ValueError(f"{name} must have value >= {minimum}, was {value}")


def validate_min_exclusive(name: str, value: float, minimum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum:
        raise ValueError(f"{name} must have value > {minimum}, was {value}")


def validate_max(name: str, value: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value > maximum:
        raise ValueError(f"{name} must have value <= {maximum}, was {value}")


def validate_max_exclusive(name: str, value: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value >= maximum:
        raise ValueError(f"{name} must have value < {maximum}, was {value}")


def validate_range(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must have value >= {minimum} and <= {maximum}, was {value}")


def validate_range_exclusive(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum or value >= maximum:
        raise ValueError(f"{name} must have value > {minimum} and < {maximum}, was {value}")


def validate_range_exclusive_min(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value <= minimum or value > maximum:
        raise ValueError(f"{name} must have value > {minimum} and <= {maximum}, was {value}")


def validate_range_exclusive_max(name: str, value: float, minimum: float, maximum: float) -> None:
    """
    Args:
        name: value name.
        value: value to check.
        minimum: minimum value allowed.
        maximum: maximum value allowed.
    Raises:
        ValueError: invalid value
    """
    if value < minimum or value >= maximum:
        raise ValueError(f"{name} must have value >= {minimum} and < {maximum}, was {value}")
