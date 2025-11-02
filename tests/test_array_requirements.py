"""Tests for automatic array conversion and requirements in PyVolGrid."""

import numpy as np

from pyvolgrid import volume_from_spheres


class TestArrayRequirements:
    """Test automatic array conversion and compatibility."""

    def test_c_contiguous_arrays_accepted(self):
        """Test that C-contiguous float64 arrays are accepted."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii = np.array([1.0], dtype=np.float64)

        # Verify arrays are C-contiguous
        assert coords.flags.c_contiguous
        assert radii.flags.c_contiguous

        # Should work without issues
        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_non_contiguous_coords_accepted(self):
        """Test that non-contiguous coordinate arrays are automatically converted."""
        # Create a non-contiguous array by slicing
        large_coords = np.array(
            [[0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 999.0]], dtype=np.float64
        )
        non_contiguous_coords = large_coords[:, :3]  # Creates non-contiguous view
        radii = np.array([1.0, 1.0], dtype=np.float64)

        # Verify the array is not C-contiguous
        assert not non_contiguous_coords.flags.c_contiguous

        # Should work fine - conversion happens automatically
        result = volume_from_spheres(non_contiguous_coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_non_contiguous_radii_accepted(self):
        """Test that non-contiguous radii arrays are automatically converted."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        # Create non-contiguous radii array
        large_radii = np.array([1.0, 999.0, 2.0, 999.0], dtype=np.float64)
        non_contiguous_radii = large_radii[::2]  # Creates non-contiguous view

        # Verify the array is not C-contiguous
        assert not non_contiguous_radii.flags.c_contiguous

        # Need to adjust coords to match the radii length
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)

        # Should work fine - conversion happens automatically
        result = volume_from_spheres(coords, non_contiguous_radii)
        assert isinstance(result, float)
        assert result > 0

    def test_ascontiguousarray_works(self):
        """Test that np.ascontiguousarray can fix non-contiguous arrays."""
        # Create non-contiguous coords
        large_coords = np.array(
            [[0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 999.0]], dtype=np.float64
        )
        non_contiguous_coords = large_coords[:, :3]
        radii = np.array([1.0, 1.0], dtype=np.float64)

        # Verify it's not contiguous
        assert not non_contiguous_coords.flags.c_contiguous

        # Fix with ascontiguousarray
        contiguous_coords = np.ascontiguousarray(non_contiguous_coords)
        assert contiguous_coords.flags.c_contiguous

        # Should work now
        result = volume_from_spheres(contiguous_coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_fortran_contiguous_accepted(self):
        """Test that Fortran-contiguous arrays are automatically converted."""
        # Create Fortran-contiguous array (need larger array to ensure it's not also C-contiguous)
        coords_f = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64, order="F"
        )
        radii = np.array([1.0, 1.0], dtype=np.float64)

        # Verify it's Fortran-contiguous
        assert coords_f.flags.f_contiguous
        # For 2x3 arrays, F-order should not be C-contiguous
        if not coords_f.flags.c_contiguous:
            # Should work fine - conversion happens automatically
            result = volume_from_spheres(coords_f, radii)
            assert isinstance(result, float)
            assert result > 0

    def test_float32_arrays_converted(self):
        """Test that float32 arrays are automatically converted to float64."""
        coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        radii = np.array([1.0], dtype=np.float32)

        # Verify arrays are contiguous but wrong dtype
        assert coords.flags.c_contiguous
        assert radii.flags.c_contiguous
        assert coords.dtype == np.float32
        assert radii.dtype == np.float32

        # Should work fine - conversion to float64 happens automatically
        result = volume_from_spheres(coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_mixed_requirements_converted(self):
        """Test that arrays with multiple issues are automatically fixed."""
        # Create float32 array that's also non-contiguous
        large_coords = np.array(
            [[0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 999.0]], dtype=np.float32
        )  # Wrong dtype
        non_contiguous_coords = large_coords[:, :3]  # Wrong layout
        radii = np.array([1.0, 1.0], dtype=np.float64)  # Correct

        # Verify multiple issues
        assert not non_contiguous_coords.flags.c_contiguous  # Wrong layout
        assert non_contiguous_coords.dtype == np.float32  # Wrong dtype

        # Should work fine - both issues are automatically fixed
        result = volume_from_spheres(non_contiguous_coords, radii)
        assert isinstance(result, float)
        assert result > 0

    def test_stride_behavior_with_conversion(self):
        """Test stride behavior when arrays are automatically converted."""
        # Create non-contiguous array
        large_coords = np.array(
            [[0.0, 0.0, 0.0, 999.0], [1.0, 1.0, 1.0, 999.0]], dtype=np.float64
        )
        non_contiguous_coords = large_coords[:, :3]
        radii = np.array([1.0, 1.0], dtype=np.float64)

        # Expected strides for C-contiguous (2, 3) float64 array would be (24, 8)
        # But our non-contiguous array has strides (32, 8) due to the extra column
        expected_contiguous_strides = (3 * 8, 8)  # (24, 8)
        actual_strides = non_contiguous_coords.strides

        assert actual_strides != expected_contiguous_strides
        assert actual_strides == (32, 8)  # Due to the 4-column source array

        # The function should now work - conversion fixes the stride issue
        result = volume_from_spheres(non_contiguous_coords, radii)
        assert isinstance(result, float)
        assert result > 0
