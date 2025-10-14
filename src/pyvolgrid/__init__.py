from importlib.metadata import version, PackageNotFoundError

import numpy as np
from numpy.typing import NDArray
from pyvolgrid._core import _volume_from_spheres

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "0.0.0"


def volume_from_spheres(
    coords: NDArray[np.float64],
    radii: NDArray[np.float64],
    grid_spacing: float = 0.1,
) -> float:
    """Calculate the volume occupied by a set of spheres using a grid-based method.
    Parameters
    ----------
    coords : NDArray[np.float64]
        An array of shape (N, 3) containing the coordinates of the centers of the spheres.
    radii : NDArray[np.float64]
        An array of shape (N,) containing the radii of the spheres.
    grid_spacing : SupportsFloat, optional
        The spacing between grid points. Default is 0.1.
    Returns
    -------
    float
        The total volume occupied by the spheres."""

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be of shape (N, 3)")
    if radii.ndim != 1 or radii.shape[0] != coords.shape[0]:
        raise ValueError(
            "radii must be of shape (N,) and match the number of coordinates"
        )
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be greater than 0.0")

    return _volume_from_spheres(coords, radii, grid_spacing)


__all__ = ["__version__", "volume_from_spheres"]
