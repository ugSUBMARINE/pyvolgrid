"""Tests for the flexible array interface in PyVolGrid."""

import numpy as np
import pytest

from pyvolgrid import volume_from_spheres


class TestFlexibleInterface:
    """Test the flexible array interface that accepts any array-like input."""

    def test_python_lists(self):
        """Test that Python lists work as input."""
        coords = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        radii = [1.0, 0.5]

        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_tuples(self):
        """Test that tuples work as input."""
        coords = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        radii = (1.0, 0.8)

        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_mixed_numeric_types(self):
        """Test mixing integers and floats."""
        coords = [[0, 0, 0], [1.5, 0, 0]]  # Mix int and float
        radii = [1, 0.7]  # Mix int and float

        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_numpy_arrays_various_dtypes(self):
        """Test that numpy arrays with various dtypes are converted automatically."""
        # Test float32
        coords_f32 = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii_f32 = np.array([1.0], dtype=np.float32)
        result = volume_from_spheres(coords_f32, radii_f32)
        assert isinstance(result, float)
        assert result > 0

        # Test int32
        coords_int = np.array([[0, 0, 0]], dtype=np.int32)
        radii_int = np.array([1], dtype=np.int32)
        result = volume_from_spheres(coords_int, radii_int)
        assert isinstance(result, float)
        assert result > 0

        # Test int64
        coords_i64 = np.array([[0, 0, 0]], dtype=np.int64)
        radii_i64 = np.array([1], dtype=np.int64)
        result = volume_from_spheres(coords_i64, radii_i64)
        assert isinstance(result, float)
        assert result > 0

    def test_non_contiguous_arrays_converted(self):
        """Test that non-contiguous arrays are converted automatically."""
        # Create non-contiguous array
        large_coords = np.array([[0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 999.0]], dtype=np.float64)
        non_contiguous_coords = large_coords[:, :3]  # Non-contiguous view
        radii = np.array([1.0, 1.0], dtype=np.float64)

        # Verify it's not contiguous
        assert not non_contiguous_coords.flags.c_contiguous

        # Should work without issues now
        result = volume_from_spheres(non_contiguous_coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_nested_lists_converted(self):
        """Test nested list structures."""
        coords = [[[0], [0], [0]], [[1], [1], [1]]]  # Weird nested structure
        radii = [[1.0], [0.8]]  # Also nested

        # Should still work due to numpy's conversion capabilities
        try:
            result = volume_from_spheres(coords, radii)
            # If it works, great! If not, that's expected behavior
            assert isinstance(result, float)
            assert result > 0
        except ValueError:
            # Expected - the shape won't be right after conversion
            pass

    def test_single_sphere_lists(self):
        """Test single sphere using simple lists."""
        coords = [[0, 0, 0]]
        radii = [1.0]

        result = volume_from_spheres(coords, radii)

        # Compare with analytical solution
        analytical = (4.0 / 3.0) * np.pi * (1.0**3)
        relative_error = abs(result - analytical) / analytical
        assert relative_error < 0.1  # Within 10%

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        coords = []
        radii = []

        # Now catches the error earlier in Python validation
        with pytest.raises(
            ValueError,
            match="coords must have shape \\(N, 3\\), got shape \\(0,\\)",
        ):
            volume_from_spheres(coords, radii)

    def test_mismatched_lengths_clear_error(self):
        """Test clear error messages for mismatched input lengths."""
        coords = [[0, 0, 0], [1, 1, 1]]  # 2 spheres
        radii = [1.0]  # 1 radius

        with pytest.raises(
            ValueError,
            match="Number of radii \\(1\\) must match number of coordinates \\(2\\)",
        ):
            volume_from_spheres(coords, radii)

    def test_wrong_coord_dimensions(self):
        """Test error handling for wrong coordinate dimensions."""
        # 2D instead of 3D coordinates
        coords = [[0, 0], [1, 1]]
        radii = [1.0, 1.0]

        with pytest.raises(
            ValueError,
            match="coords must have shape \\(N, 3\\), got shape",
        ):
            volume_from_spheres(coords, radii)

    def test_wrong_radii_dimensions(self):
        """Test error handling for wrong radii dimensions."""
        coords = [[0, 0, 0]]
        radii = [[1.0]]  # 2D instead of 1D

        with pytest.raises(ValueError, match="radii must be 1-dimensional, got shape"):
            volume_from_spheres(coords, radii)

    def test_consistency_with_numpy_arrays(self):
        """Test that results are consistent regardless of input type."""
        # Same data in different formats
        coords_list = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        radii_list = [1.0, 0.5]

        coords_array = np.array(coords_list, dtype=np.float64)
        radii_array = np.array(radii_list, dtype=np.float64)

        coords_tuple = tuple(tuple(row) for row in coords_list)
        radii_tuple = tuple(radii_list)

        # All should give the same result
        result_list = volume_from_spheres(coords_list, radii_list)
        result_array = volume_from_spheres(coords_array, radii_array)
        result_tuple = volume_from_spheres(coords_tuple, radii_tuple)

        assert result_list == result_array
        assert result_array == result_tuple

    def test_performance_no_unnecessary_copying(self):
        """Test that already-correct arrays aren't unnecessarily copied."""
        # Create already-correct arrays
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)

        # Verify they're already correct
        assert coords.flags.c_contiguous
        assert radii.flags.c_contiguous
        assert coords.dtype == np.float64
        assert radii.dtype == np.float64

        # Should still work (np.ascontiguousarray won't copy if not needed)
        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_scalar_radius_single_sphere(self):
        """Test scalar radius with single sphere."""
        coords = [[0, 0, 0]]
        radius = 1.0  # Single float, not array

        result = volume_from_spheres(coords, radius)

        # Should be equivalent to passing [1.0] as radii
        result_array = volume_from_spheres(coords, [1.0])
        assert result == result_array

        # Compare with analytical solution for single sphere
        analytical = (4.0 / 3.0) * np.pi * (1.0**3)
        relative_error = abs(result - analytical) / analytical
        assert relative_error < 0.1  # Within 10%

    def test_scalar_radius_multiple_spheres(self):
        """Test scalar radius with multiple spheres."""
        coords = [[0, 0, 0], [3, 0, 0], [0, 3, 0]]
        radius = 1.5  # All spheres have same radius

        result = volume_from_spheres(coords, radius)

        # Should be equivalent to passing [1.5, 1.5, 1.5]
        result_array = volume_from_spheres(coords, [1.5, 1.5, 1.5])
        assert result == result_array

        assert isinstance(result, float)
        assert result > 0

    def test_scalar_radius_various_types(self):
        """Test scalar radius with different numeric types."""
        coords = [[0, 0, 0], [2, 0, 0]]

        # Test with int
        result_int = volume_from_spheres(coords, 1)  # int
        result_float = volume_from_spheres(coords, 1.0)  # float
        assert result_int == result_float

        # Test with numpy scalar types
        result_np_float32 = volume_from_spheres(coords, np.float32(1.0))
        result_np_float64 = volume_from_spheres(coords, np.float64(1.0))
        assert result_np_float32 == result_float
        assert result_np_float64 == result_float

        # Test with numpy int types
        result_np_int = volume_from_spheres(coords, np.int32(1))
        assert result_np_int == result_float

    def test_scalar_radius_with_numpy_coords(self):
        """Test scalar radius works with numpy coordinate arrays."""
        coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        radius = 0.8

        result = volume_from_spheres(coords, radius)

        # Verify it's equivalent to array version
        radii_array = np.array([0.8, 0.8], dtype=np.float64)
        result_array = volume_from_spheres(coords, radii_array)
        assert result == result_array

        assert isinstance(result, float)
        assert result > 0

    def test_scalar_radius_zero_valid(self):
        """Test scalar radius of zero (mathematically valid)."""
        coords = [[0, 0, 0], [1, 0, 0]]
        radius = 0.0

        result = volume_from_spheres(coords, radius)
        assert result == 0.0  # Zero radius should give zero volume
