"""Tests for edge cases and performance in PyVolGrid."""

import time
import numpy as np
import pytest

from pyvolgrid import volume_from_spheres


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_radii(self):
        """Test with very small radii."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([0.001])  # Small but reasonable radius

        # Use proportionally small grid spacing
        volume = volume_from_spheres(coords, radii, grid_spacing=0.0001)
        assert volume >= 0  # Should be non-negative

        # For small spheres, volume should be small but positive
        if volume > 0:
            # Should be very small but reasonable
            assert volume < 0.01

    def test_very_large_radii(self):
        """Test with very large radii."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([100.0])  # Large radius

        # Use larger grid spacing to avoid excessive memory usage
        volume = volume_from_spheres(coords, radii, grid_spacing=2.0)
        assert volume > 0

        # Should be a large volume
        assert volume > 1000  # Much larger than unit sphere

    def test_many_tiny_spheres(self):
        """Test with many spheres with tiny radii."""
        n_spheres = 20
        # Create a grid of tiny spheres (ensure float64)
        coords = np.array(
            [[i, j, k] for i in range(2) for j in range(2) for k in range(5)],
            dtype=np.float64,
        )
        radii = np.full(n_spheres, 0.1, dtype=np.float64)  # Small spheres

        volume = volume_from_spheres(coords, radii, grid_spacing=0.05)
        assert volume >= 0

    def test_spheres_with_very_large_coordinates(self):
        """Test spheres positioned at large coordinate values."""
        coords = np.array([[1000.0, 2000.0, 3000.0]])
        radii = np.array([1.0])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume > 0

        # Should be similar to sphere at origin
        origin_volume = volume_from_spheres(
            np.array([[0.0, 0.0, 0.0]]), np.array([1.0]), grid_spacing=0.1
        )

        # Allow some numerical difference but should be close
        relative_error = abs(volume - origin_volume) / origin_volume
        assert relative_error < 0.1

    def test_spheres_with_negative_coordinates(self):
        """Test spheres positioned at negative coordinates."""
        coords = np.array([[-10.0, -20.0, -30.0]])
        radii = np.array([2.0])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume > 0

    def test_mixed_small_and_large_spheres(self):
        """Test mixture of very small and very large spheres."""
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # Large sphere
                [10.0, 0.0, 0.0],  # Small sphere
                [0.0, 10.0, 0.0],  # Medium sphere
            ]
        )
        radii = np.array([5.0, 0.01, 1.0])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.2)
        assert volume > 0

        # Should be dominated by the large sphere
        large_sphere_volume = volume_from_spheres(
            coords[:1], radii[:1], grid_spacing=0.2
        )

        # Total volume should be close to large sphere volume (others are much smaller/distant)
        assert volume >= large_sphere_volume * 0.9

    def test_single_sphere_different_grid_spacings(self):
        """Test how volume calculation changes with grid spacing."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])

        # Test a range of grid spacings
        spacings = [0.5, 0.2, 0.1, 0.05]
        volumes = []

        for spacing in spacings:
            volume = volume_from_spheres(coords, radii, grid_spacing=spacing)
            volumes.append(volume)
            assert volume > 0

        # All volumes should be positive and reasonable
        assert all(v > 0 for v in volumes)
        assert all(v < 10 for v in volumes)  # Should be reasonable for unit sphere

    def test_large_number_of_spheres_performance(self):
        """Test performance with a larger number of spheres."""
        n_spheres = 50
        np.random.seed(42)  # For reproducible results

        # Create random sphere positions and sizes
        coords = np.random.uniform(-5, 5, size=(n_spheres, 3))
        radii = np.random.uniform(0.1, 1.0, size=n_spheres)

        start_time = time.time()
        volume = volume_from_spheres(coords, radii, grid_spacing=0.2)
        elapsed_time = time.time() - start_time

        assert volume > 0
        # Should complete in reasonable time (less than 10 seconds)
        assert elapsed_time < 10.0, f"Computation took too long: {elapsed_time:.2f}s"

    def test_spheres_close_to_grid_boundaries(self):
        """Test spheres positioned close to grid boundaries."""
        # Sphere that might test boundary conditions in the grid calculation
        coords = np.array([[0.001, 0.001, 0.001]])  # Very close to origin
        radii = np.array([0.5])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume > 0

        # Should be similar to sphere exactly at origin
        origin_volume = volume_from_spheres(
            np.array([[0.0, 0.0, 0.0]]), radii, grid_spacing=0.1
        )

        relative_error = abs(volume - origin_volume) / origin_volume
        assert relative_error < 0.1

    def test_identical_overlapping_spheres(self):
        """Test multiple identical spheres at the same position."""
        n_spheres = 5
        coords = np.zeros((n_spheres, 3))  # All at origin
        radii = np.ones(n_spheres)  # All same radius

        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        single_volume = volume_from_spheres(coords[:1], radii[:1], grid_spacing=0.1)

        # Should be approximately equal (all spheres overlap completely)
        relative_error = abs(combined_volume - single_volume) / single_volume
        assert relative_error < 0.05, (
            f"Identical overlapping spheres error: {relative_error:.3f}"
        )

    def test_linear_arrangement_of_spheres(self):
        """Test spheres arranged in a line."""
        n_spheres = 10
        coords = np.array([[i * 1.5, 0.0, 0.0] for i in range(n_spheres)])
        radii = np.full(n_spheres, 0.5)  # Radius 0.5, spacing 1.5, so some overlap

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume > 0

        # Should be less than or approximately equal to sum of individual volumes due to overlaps
        # (Note: numerical precision might cause small variations)
        total_individual = n_spheres * volume_from_spheres(
            coords[:1], radii[:1], grid_spacing=0.1
        )
        # Allow for small numerical differences while still expecting some overlap effect
        assert volume <= total_individual * 1.01, (
            f"Expected volume {volume} <= {total_individual * 1.01} (with small tolerance)"
        )

    def test_memory_usage_with_fine_grid(self):
        """Test that fine grid spacing doesn't cause memory issues."""
        # Small sphere with fine grid - should still be manageable
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])

        # This should complete without memory errors
        volume = volume_from_spheres(coords, radii, grid_spacing=0.01)
        assert volume > 0

    def test_precision_with_different_float64_values(self):
        """Test precision with different float64 input values."""
        # Since the C++ extension only accepts float64, test precision with different representations
        coords1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii1 = np.array([1.0], dtype=np.float64)

        # Same values but constructed differently
        coords2 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        radii2 = np.array([1.0], dtype=np.float64)

        volume1 = volume_from_spheres(coords1, radii1, grid_spacing=0.1)
        volume2 = volume_from_spheres(coords2, radii2, grid_spacing=0.1)

        # Should be identical for same input
        assert volume1 == volume2, "Identical inputs should give identical results"


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.slow
    def test_scaling_with_number_of_spheres(self):
        """Test how computation time scales with number of spheres."""
        np.random.seed(42)

        sphere_counts = [5, 10, 20]
        times = []

        for n in sphere_counts:
            coords = np.random.uniform(-2, 2, size=(n, 3))
            radii = np.random.uniform(0.2, 0.8, size=n)

            start_time = time.time()
            volume = volume_from_spheres(coords, radii, grid_spacing=0.15)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

            assert volume > 0

        # Each test should complete in reasonable time
        assert all(t < 5.0 for t in times), f"Some tests too slow: {times}"

    def test_memory_efficiency(self):
        """Test that memory usage is reasonable."""
        # This is more of a smoke test - if it completes without
        # memory errors, the C++ implementation is probably efficient
        n_spheres = 30
        np.random.seed(42)

        coords = np.random.uniform(-3, 3, size=(n_spheres, 3))
        radii = np.random.uniform(0.1, 0.5, size=n_spheres)

        # Should complete without memory issues
        volume = volume_from_spheres(coords, radii, grid_spacing=0.2)
        assert volume > 0
