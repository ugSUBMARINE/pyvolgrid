"""Common test fixtures and utilities for PyVolGrid tests."""

import math
import numpy as np
import pytest


@pytest.fixture
def single_sphere():
    """Fixture for a single sphere at origin with radius 1."""
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    radii = np.array([1.0], dtype=np.float64)
    return coords, radii


@pytest.fixture
def two_non_overlapping_spheres():
    """Fixture for two non-overlapping spheres."""
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float64)
    radii = np.array([1.0, 1.0], dtype=np.float64)
    return coords, radii


@pytest.fixture
def two_overlapping_spheres():
    """Fixture for two overlapping spheres."""
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    radii = np.array([1.0, 1.0], dtype=np.float64)
    return coords, radii


@pytest.fixture
def analytical_sphere_volume():
    """Fixture that returns a function to calculate analytical sphere volume."""

    def volume_func(radius):
        return (4.0 / 3.0) * math.pi * (radius**3)

    return volume_func


@pytest.fixture
def default_grid_spacing():
    """Default grid spacing for tests."""
    return 0.1


def assert_relative_error(actual, expected, max_error=0.1, message=""):
    """Helper function to assert relative error is within bounds."""
    if expected == 0:
        assert actual == 0, f"Expected zero but got {actual}. {message}"
    else:
        relative_error = abs(actual - expected) / abs(expected)
        assert relative_error <= max_error, (
            f"Relative error {relative_error:.4f} exceeds {max_error}. "
            f"Actual: {actual}, Expected: {expected}. {message}"
        )


def create_random_spheres(
    n_spheres, seed=42, coord_range=(-2, 2), radius_range=(0.1, 1.0)
):
    """Helper function to create random spheres for testing."""
    np.random.seed(seed)
    coords = np.random.uniform(coord_range[0], coord_range[1], size=(n_spheres, 3))
    radii = np.random.uniform(radius_range[0], radius_range[1], size=n_spheres)
    return coords.astype(np.float64), radii.astype(np.float64)
