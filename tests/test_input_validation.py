"""Tests for input validation in PyVolGrid."""

import numpy as np
import pytest

from pyvolgrid import volume_from_spheres


class TestInputValidation:
    """Test input validation for the volume_from_spheres function."""

    def test_valid_input_single_sphere(self):
        """Test that valid input for a single sphere works."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        result = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert isinstance(result, float)
        assert result > 0

    def test_valid_input_multiple_spheres(self):
        """Test that valid input for multiple spheres works."""
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 0.5])
        result = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert isinstance(result, float)
        assert result > 0

    def test_coords_wrong_dimensions_1d(self):
        """Test that 1D coords array raises ValueError."""
        coords = np.array([0.0, 0.0, 0.0])  # Should be (1, 3)
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="coords must be of shape \\(N, 3\\)"):
            volume_from_spheres(coords, radii)

    def test_coords_wrong_dimensions_3d(self):
        """Test that 3D coords array raises ValueError."""
        coords = np.array([[[0.0, 0.0, 0.0]]])  # Should be (1, 3)
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="coords must be of shape \\(N, 3\\)"):
            volume_from_spheres(coords, radii)

    def test_coords_wrong_second_dimension(self):
        """Test that coords with wrong second dimension raises ValueError."""
        coords = np.array([[0.0, 0.0]])  # Should have 3 columns, not 2
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="coords must be of shape \\(N, 3\\)"):
            volume_from_spheres(coords, radii)

    def test_coords_four_dimensions(self):
        """Test that coords with 4 dimensions instead of 3 raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0, 0.0]])  # Should have 3 columns, not 4
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="coords must be of shape \\(N, 3\\)"):
            volume_from_spheres(coords, radii)

    def test_radii_wrong_dimensions_2d(self):
        """Test that 2D radii array raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([[1.0]])  # Should be 1D

        with pytest.raises(
            ValueError,
            match="radii must be of shape \\(N,\\) and match the number of coordinates",
        ):
            volume_from_spheres(coords, radii)

    def test_radii_wrong_dimensions_0d(self):
        """Test that 0D radii array raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array(1.0)  # Should be 1D array, not scalar

        with pytest.raises(
            ValueError,
            match="radii must be of shape \\(N,\\) and match the number of coordinates",
        ):
            volume_from_spheres(coords, radii)

    def test_mismatched_array_sizes(self):
        """Test that mismatched coords and radii sizes raise ValueError."""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # 2 spheres
        radii = np.array([1.0])  # Only 1 radius

        with pytest.raises(
            ValueError,
            match="radii must be of shape \\(N,\\) and match the number of coordinates",
        ):
            volume_from_spheres(coords, radii)

    def test_more_radii_than_coords(self):
        """Test that more radii than coords raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0]])  # 1 sphere
        radii = np.array([1.0, 0.5])  # 2 radii

        with pytest.raises(
            ValueError,
            match="radii must be of shape \\(N,\\) and match the number of coordinates",
        ):
            volume_from_spheres(coords, radii)

    def test_zero_grid_spacing(self):
        """Test that zero grid spacing raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="grid_spacing must be greater than 0.0"):
            volume_from_spheres(coords, radii, grid_spacing=0.0)

    def test_negative_grid_spacing(self):
        """Test that negative grid spacing raises ValueError."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])

        with pytest.raises(ValueError, match="grid_spacing must be greater than 0.0"):
            volume_from_spheres(coords, radii, grid_spacing=-0.1)

    def test_empty_arrays(self):
        """Test behavior with empty arrays."""
        coords = np.array([]).reshape(0, 3)
        radii = np.array([])
        # The C++ implementation currently has an issue with empty arrays
        # This should be handled gracefully in future versions
        with pytest.raises(
            ValueError, match="Cannot find the maximum of an empty array"
        ):
            volume_from_spheres(coords, radii)

    def test_numpy_array_types(self):
        """Test that function requires float64 arrays."""
        # The C++ extension requires float64 arrays specifically

        # Test with float32 - should fail
        coords_f32 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii_f32 = np.array([1.0], dtype=np.float32)
        with pytest.raises(TypeError):
            volume_from_spheres(coords_f32, radii_f32)

        # Test with int - should fail
        coords_int = np.array([[0, 0, 0]], dtype=np.int32)
        radii_int = np.array([1], dtype=np.int32)
        with pytest.raises(TypeError):
            volume_from_spheres(coords_int, radii_int)

        # Test with explicit float64 - should work
        coords_f64 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii_f64 = np.array([1.0], dtype=np.float64)
        result = volume_from_spheres(coords_f64, radii_f64)
        assert isinstance(result, float)
        assert result > 0

    def test_very_small_positive_grid_spacing(self):
        """Test with very small but positive grid spacing."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        # Use a small but reasonable grid spacing to avoid memory issues
        result = volume_from_spheres(coords, radii, grid_spacing=0.01)
        assert isinstance(result, float)
        assert result > 0
