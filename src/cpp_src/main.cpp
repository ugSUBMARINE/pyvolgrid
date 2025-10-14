#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "volgrid.hpp"

namespace py = pybind11;

double calc_vol(py::array_t<double> coords, py::array_t<double> radii, double grid_spacing) {
    py::buffer_info buf1 = coords.request();
    py::buffer_info buf2 = radii.request();

    // Basic validation of input shapes is done in Python before calling this function.
    // Uncomment the following lines for additional safety checks in C++:
    // if (buf1.ndim != 2 || buf1.shape[1] != 3) {
    //     throw std::runtime_error("Coordinates array must be of shape (N, 3).");
    // }
    // if (buf2.ndim != 1 || buf2.shape[0] != buf1.shape[0]) {
    //     throw std::runtime_error("Radii array must be 1D and match number of coordinates.");
    // }

    double* ptr_coords = static_cast<double*>(buf1.ptr);
    double* ptr_radii = static_cast<double*>(buf2.ptr);
    size_t n_spheres = buf1.shape[0];

    double volume = 0.0;
    {
        py::gil_scoped_release release;
        volume = volume_of_spheres(ptr_coords, ptr_radii, n_spheres, grid_spacing);
    }

    return volume;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "C++ extension for volume calculation using a grid-based approach";

  m.def("_volume_from_spheres", &calc_vol, R"pbdoc(
        Calculate the volume occupied by spheres using a grid-based approach.

        Args:
            coords: N x 3 array of sphere center coordinates (float64)
            radii: 1D array of sphere radii (float64, length N)
            grid_spacing: Grid spacing for the volume calculation (float64)

        Returns:
            float: Estimated volume occupied by the spheres

        Raises:
            RuntimeError: If input arrays have incorrect shapes or sizes
  )pbdoc",
  py::arg("coords").noconvert(), py::arg("radii").noconvert(), py::arg("grid_spacing")
  );
}
