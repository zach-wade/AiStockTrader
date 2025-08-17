"""
Secure Random Number Generation Utilities

This module provides cryptographically secure random number generation
to replace insecure random usage in financial calculations.

Created to address G2.4 Insecure Random Number Generation vulnerability.
"""

# Standard library imports
import logging
import secrets

# Third-party imports
import numpy as np

logger = logging.getLogger(__name__)


class SecureRandom:
    """
    Cryptographically secure random number generator for financial calculations.

    Uses the secrets module which provides access to the most secure source of
    randomness that the operating system provides.
    """

    def __init__(self):
        self._system_random = secrets.SystemRandom()

    def uniform(self, low: float, high: float) -> float:
        """
        Generate a cryptographically secure random float between low and high.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            Cryptographically secure random float
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")

        return self._system_random.uniform(low, high)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """
        Generate a cryptographically secure random number from normal distribution.

        Uses Box-Muller transform with secure uniform random numbers.

        Args:
            mu: Mean of the distribution
            sigma: Standard deviation of the distribution

        Returns:
            Cryptographically secure random float from normal distribution
        """
        # Box-Muller transform using secure random numbers
        u1 = self._system_random.uniform(0, 1)
        u2 = self._system_random.uniform(0, 1)

        # Avoid log(0) by ensuring u1 is not exactly 0
        while u1 == 0:
            u1 = self._system_random.uniform(0, 1)

        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return mu + sigma * z0

    def randint(self, low: int, high: int) -> int:
        """
        Generate a cryptographically secure random integer between low and high.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)

        Returns:
            Cryptographically secure random integer
        """
        if low >= high:
            raise ValueError(f"low ({low}) must be less than high ({high})")

        return self._system_random.randrange(low, high)

    def choice(self, sequence: list) -> int | float | str:
        """
        Choose a cryptographically secure random element from a sequence.

        Args:
            sequence: List or sequence to choose from

        Returns:
            Randomly chosen element
        """
        if not sequence:
            raise ValueError("Cannot choose from empty sequence")

        return self._system_random.choice(sequence)

    def shuffle(self, sequence: list) -> list:
        """
        Shuffle a sequence using cryptographically secure randomness.

        Args:
            sequence: List to shuffle

        Returns:
            New list with elements shuffled
        """
        shuffled = sequence.copy()
        self._system_random.shuffle(shuffled)
        return shuffled

    def sample(self, population: list, k: int) -> list:
        """
        Sample k elements from population without replacement using secure randomness.

        Args:
            population: Population to sample from
            k: Number of elements to sample

        Returns:
            List of k randomly sampled elements
        """
        if k > len(population):
            raise ValueError("Sample size cannot be larger than population")

        return self._system_random.sample(population, k)

    def secure_numpy_uniform(
        self, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | np.ndarray:
        """
        Generate cryptographically secure uniform random numbers compatible with numpy.

        Args:
            low: Lower bound (inclusive)
            high: Upper bound (exclusive)
            size: Output shape. If None, returns a single float

        Returns:
            Secure random float or numpy array
        """
        if size is None:
            return self.uniform(low, high)

        if isinstance(size, int):
            return np.array([self.uniform(low, high) for _ in range(size)])

        # Handle tuple shapes
        total_elements = np.prod(size)
        flat_array = np.array([self.uniform(low, high) for _ in range(total_elements)])
        return flat_array.reshape(size)

    def secure_numpy_normal(
        self, loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | np.ndarray:
        """
        Generate cryptographically secure normal random numbers compatible with numpy.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
            size: Output shape. If None, returns a single float

        Returns:
            Secure random float or numpy array from normal distribution
        """
        if size is None:
            return self.normal(loc, scale)

        if isinstance(size, int):
            return np.array([self.normal(loc, scale) for _ in range(size)])

        # Handle tuple shapes
        total_elements = np.prod(size)
        flat_array = np.array([self.normal(loc, scale) for _ in range(total_elements)])
        return flat_array.reshape(size)


# Global secure random instance
_secure_random = SecureRandom()


# Convenience functions for direct usage
def secure_uniform(low: float, high: float) -> float:
    """Generate cryptographically secure uniform random float."""
    return _secure_random.uniform(low, high)


def secure_normal(mu: float = 0.0, sigma: float = 1.0) -> float:
    """Generate cryptographically secure normal random float."""
    return _secure_random.normal(mu, sigma)


def secure_randint(low: int, high: int) -> int:
    """Generate cryptographically secure random integer."""
    return _secure_random.randint(low, high)


def secure_choice(sequence: list) -> int | float | str:
    """Choose cryptographically secure random element from sequence."""
    return _secure_random.choice(sequence)


def secure_sample(population: list, k: int) -> list:
    """Sample k elements from population without replacement using secure randomness."""
    return _secure_random.sample(population, k)


def secure_shuffle(sequence: list) -> None:
    """Shuffle a sequence in-place using cryptographically secure randomness."""
    _secure_random._system_random.shuffle(sequence)


def secure_numpy_uniform(
    low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
) -> float | np.ndarray:
    """Generate cryptographically secure uniform random numbers (numpy compatible)."""
    return _secure_random.secure_numpy_uniform(low, high, size)


def secure_numpy_normal(
    loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None
) -> float | np.ndarray:
    """Generate cryptographically secure normal random numbers (numpy compatible)."""
    return _secure_random.secure_numpy_normal(loc, scale, size)


def get_secure_random() -> SecureRandom:
    """Get the global secure random instance."""
    return _secure_random


# Migration helpers for replacing insecure random usage
class SecureRandomMigrationHelper:
    """
    Helper class to ease migration from insecure random usage.

    Provides drop-in replacements for common insecure random patterns.
    """

    @staticmethod
    def replace_random_uniform(low: float, high: float) -> float:
        """
        Secure replacement for secure_uniform().

        SECURITY FIX: Replaces non-cryptographic secure_uniform()
        with cryptographically secure alternative.
        """
        logger.info(f"SECURITY: Using secure random instead of secure_uniform({low}, {high})")
        return secure_uniform(low, high)

    @staticmethod
    def replace_numpy_random_uniform(
        low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | np.ndarray:
        """
        Secure replacement for np.secure_uniform().

        SECURITY FIX: Replaces non-cryptographic np.secure_uniform()
        with cryptographically secure alternative.
        """
        logger.info(
            f"SECURITY: Using secure random instead of np.secure_uniform({low}, {high}, {size})"
        )
        return secure_numpy_uniform(low, high, size)

    @staticmethod
    def replace_numpy_random_normal(
        loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None
    ) -> float | np.ndarray:
        """
        Secure replacement for secure_numpy_normal().

        SECURITY FIX: Replaces non-cryptographic secure_numpy_normal()
        with cryptographically secure alternative.
        """
        logger.info(
            f"SECURITY: Using secure random instead of secure_numpy_normal({loc}, {scale}, {size})"
        )
        return secure_numpy_normal(loc, scale, size)


# Create global migration helper instance
migration_helper = SecureRandomMigrationHelper()

# Aliases for easy migration
secure_replace_random_uniform = migration_helper.replace_random_uniform
secure_replace_numpy_uniform = migration_helper.replace_numpy_random_uniform
secure_replace_numpy_normal = migration_helper.replace_numpy_random_normal
