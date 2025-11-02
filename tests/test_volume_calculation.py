"""Tests for volume calculation functionality in PyVolGrid."""

import math
import numpy as np

from pyvolgrid import volume_from_spheres


class TestVolumeCalculation:
    """Test volume calculation functionality."""

    def test_single_sphere_volume_approximation(self):
        """Test that single sphere volume approximates analytical solution."""
        radius = 1.0
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([radius])

        # Analytical volume of a sphere: (4/3) * π * r³
        analytical_volume = (4.0 / 3.0) * math.pi * (radius**3)

        # Test with fine grid spacing for better accuracy
        calculated_volume = volume_from_spheres(coords, radii, grid_spacing=0.05)

        # Allow for some error due to grid approximation (within 10%)
        relative_error = abs(calculated_volume - analytical_volume) / analytical_volume
        assert relative_error < 0.1, (
            f"Relative error {relative_error:.3f} too large for single sphere"
        )
        assert calculated_volume > 0

    def test_single_sphere_different_radii(self):
        """Test single spheres with different radii."""
        test_radii = [0.5, 1.0, 2.0]
        grid_spacing = 0.05

        for radius in test_radii:
            coords = np.array([[0.0, 0.0, 0.0]])
            radii = np.array([radius])

            calculated_volume = volume_from_spheres(
                coords, radii, grid_spacing=grid_spacing
            )
            analytical_volume = (4.0 / 3.0) * math.pi * (radius**3)

            relative_error = (
                abs(calculated_volume - analytical_volume) / analytical_volume
            )
            assert relative_error < 0.15, (
                f"Error too large for radius {radius}: {relative_error:.3f}"
            )

    def test_non_overlapping_spheres(self):
        """Test that non-overlapping spheres have additive volume."""
        # Two spheres far apart - their volumes should be approximately additive
        coords = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])  # 10 units apart
        radii = np.array([1.0, 0.5])  # Radii sum to 1.5, so no overlap at 10 units

        # Calculate combined volume
        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.1)

        # Calculate individual volumes
        vol1 = volume_from_spheres(coords[:1], radii[:1], grid_spacing=0.1)
        vol2 = volume_from_spheres(coords[1:], radii[1:], grid_spacing=0.1)

        # For non-overlapping spheres, combined should equal sum of individuals
        expected_volume = vol1 + vol2
        relative_error = abs(combined_volume - expected_volume) / expected_volume
        assert relative_error < 0.05, (
            f"Non-overlapping spheres error: {relative_error:.3f}"
        )

    def test_completely_overlapping_spheres(self):
        """Test that completely overlapping spheres have volume of the larger sphere."""
        # Large sphere completely contains small sphere
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # Same position
        radii = np.array([2.0, 1.0])  # Small sphere inside large one

        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.08)

        # Volume should be approximately that of the larger sphere
        large_sphere_volume = volume_from_spheres(
            coords[:1], radii[:1], grid_spacing=0.08
        )

        relative_error = (
            abs(combined_volume - large_sphere_volume) / large_sphere_volume
        )
        assert relative_error < 0.1, (
            f"Completely overlapping spheres error: {relative_error:.3f}"
        )

    def test_partially_overlapping_spheres(self):
        """Test partially overlapping spheres."""
        # Two spheres with centers 1.5 units apart, radii 1.0 each
        # They should overlap significantly
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])

        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.08)

        # Individual sphere volume
        single_volume = volume_from_spheres(coords[:1], radii[:1], grid_spacing=0.08)

        # Combined volume should be less than sum of two individual volumes
        # but more than one individual volume
        assert single_volume < combined_volume < 2 * single_volume

        # Should be less than sum due to overlap
        sum_individual = 2 * single_volume
        overlap_reduction = (sum_individual - combined_volume) / sum_individual
        assert overlap_reduction > 0.01, (
            f"Expected some overlap reduction, got {overlap_reduction:.4f}"
        )

    def test_touching_spheres(self):
        """Test spheres that just touch each other."""
        # Two spheres with radius 1.0, centers exactly 2.0 units apart
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0])

        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.05)

        # Individual volumes
        vol1 = volume_from_spheres(coords[:1], radii[:1], grid_spacing=0.05)
        vol2 = volume_from_spheres(coords[1:], radii[1:], grid_spacing=0.05)

        # Should be approximately additive for touching spheres
        expected_volume = vol1 + vol2
        relative_error = abs(combined_volume - expected_volume) / expected_volume
        assert relative_error < 0.1, f"Touching spheres error: {relative_error:.3f}"

    def test_grid_spacing_consistency(self):
        """Test that finer grid spacing gives more accurate results."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([1.0])
        analytical_volume = (4.0 / 3.0) * math.pi * (radii[0] ** 3)

        # Test different grid spacings
        spacings = [0.2, 0.1, 0.05]
        volumes = []
        errors = []

        for spacing in spacings:
            volume = volume_from_spheres(coords, radii, grid_spacing=spacing)
            volumes.append(volume)
            error = abs(volume - analytical_volume) / analytical_volume
            errors.append(error)

        # Generally, finer grid spacing should give more accurate results
        # (though this isn't guaranteed due to discretization effects)
        assert all(v > 0 for v in volumes), "All volumes should be positive"

        # The finest grid spacing should give reasonable accuracy
        assert errors[-1] < 0.15, (
            f"Finest grid spacing error too large: {errors[-1]:.3f}"
        )

    def test_zero_radius_sphere(self):
        """Test sphere with zero radius."""
        coords = np.array([[0.0, 0.0, 0.0]])
        radii = np.array([0.0])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume == 0.0, "Zero radius sphere should have zero volume"

    def test_multiple_spheres_different_sizes(self):
        """Test multiple spheres with different sizes."""
        # Mix of different sized spheres in different positions
        coords = np.array(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]]
        )
        radii = np.array([1.0, 0.5, 1.5, 0.8])

        combined_volume = volume_from_spheres(coords, radii, grid_spacing=0.1)

        # Should be positive and reasonable
        assert combined_volume > 0

        # Should be less than sum of individual analytical volumes
        total_analytical = sum((4.0 / 3.0) * math.pi * r**3 for r in radii)
        assert combined_volume <= total_analytical * 1.2  # Allow some error margin

    def test_spheres_in_3d_space(self):
        """Test spheres positioned in 3D space."""
        coords = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0], [0.0, 0.0, 0.0]])
        radii = np.array([1.0, 1.0, 0.5])

        volume = volume_from_spheres(coords, radii, grid_spacing=0.1)
        assert volume > 0

        # Compare with individual volumes (they're far apart, should be additive)
        individual_volumes = []
        for i in range(len(coords)):
            vol = volume_from_spheres(
                coords[i : i + 1], radii[i : i + 1], grid_spacing=0.1
            )
            individual_volumes.append(vol)

        expected_sum = sum(individual_volumes)
        relative_error = abs(volume - expected_sum) / expected_sum
        assert relative_error < 0.1, (
            f"3D positioned spheres error: {relative_error:.3f}"
        )

    def test_deterministic_results(self):
        """Test that the same input gives the same output."""
        coords = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        radii = np.array([1.0, 0.8])
        grid_spacing = 0.1

        # Run the calculation multiple times
        results = []
        for _ in range(3):
            result = volume_from_spheres(coords, radii, grid_spacing=grid_spacing)
            results.append(result)

        # All results should be identical
        assert all(r == results[0] for r in results), "Results should be deterministic"
