"""Contains fairness metrics."""

import jax.numpy as jnp


def gini(array):
    """Compute the Gini coefficient of a numpy array.

    The Gini coefficient is a measure of statistical dispersion that
    represents the income or wealth distribution of a nation's residents,
    and is commonly used as a measure of inequality.

    Values range from 0 (perfect equality) to 1 (perfect inequality).

    Args:
        array: A 1D array of values (e.g., incomes, wealth).

    Returns:
        The Gini coefficient, a float between 0 and 1.
    """
    # Sort the array
    x = jnp.sort(array)
    # Number of elements
    n = x.size
    # Cumulative sum of the sorted array
    cum = jnp.cumsum(x)
    # Relative mean absolute difference
    mad = (2 / n) * jnp.sum((jnp.arange(1, n + 1) - 0.5) * x) / cum[-1] - (n + 1) / n
    return mad
