from importlib.metadata import version, PackageNotFoundError

import numpy as np
from numpy.typing import ArrayLike
from pyvolgrid._core import _volume_from_spheres_float32, _volume_from_spheres_float64

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"


def volume_from_spheres(
    coords: ArrayLike,
    radii: ArrayLike | float,
    grid_spacing: float = 0.1,
) -> float:
    """Calculate the volume occupied by a set of spheres using a grid-based method.

    Parameters
    ----------
    coords : array-like
        An array-like object of shape (N, 3) containing the coordinates of the centers.
    radii : array-like or float
        An array-like object of shape (N,) containing the radii of the spheres.
        If a single float is provided, all spheres are assumed to have the same radius.
    grid_spacing : float, optional
        The spacing between grid points. Default is 0.1.

    Returns
    -------
    float
        The total volume occupied by the spheres.

    Notes
    -----
    Input arrays are automatically converted to C-contiguous numpy arrays as required
    by the underlying C++ implementation. Float32 and float64 dtypes are preserved
    to avoid unnecessary conversions; all other types are converted to float64.
    """

    # Convert to numpy arrays for inspection
    coords_arr = np.asarray(coords)
    radii_is_scalar = np.isscalar(radii)

    # Shape validation for coords
    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(
            f"coords must have shape (N, 3), got shape {coords_arr.shape}"
        )
    if coords_arr.shape[0] == 0:
        raise ValueError("coords must contain at least one coordinate")

    # Determine target dtype based on input dtypes
    # If coords is float32 and radii is either float32 or scalar, use float32 path
    # Otherwise use float64 path
    use_float32 = coords_arr.dtype == np.float32

    if not radii_is_scalar:
        radii_arr = np.asarray(radii)
        # Only use float32 if both coords and radii are float32
        if use_float32 and radii_arr.dtype != np.float32:
            use_float32 = False

    # Validate grid spacing
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be greater than 0.0")

    DTYPE = np.float32 if use_float32 else np.float64
    coords_array = np.ascontiguousarray(coords_arr, dtype=DTYPE)

    if radii_is_scalar:
        radii_array = np.full(coords_array.shape[0], float(radii), dtype=DTYPE)
    else:
        radii_array = np.ascontiguousarray(radii_arr, dtype=DTYPE)

    # Validate radii
    if radii_array.ndim != 1:
        raise ValueError(
            f"radii must be 1-dimensional, got shape {radii_array.shape}"
        )
    if radii_array.shape[0] != coords_array.shape[0]:
        raise ValueError(
            f"Number of radii ({radii_array.shape[0]}) must match number of coordinates ({coords_array.shape[0]})"
        )

    if use_float32:
        return _volume_from_spheres_float32(coords_array, radii_array, grid_spacing)
    else:
        return _volume_from_spheres_float64(coords_array, radii_array, grid_spacing)


__all__ = ["volume_from_spheres"]
