#ifndef VOLGRID_HPP
#define VOLGRID_HPP

#include <cstddef>

// Define a struct to hold 3D integer coordinates or indexes
struct TRi {
    int x, y, z;
};

// Define a templated struct to hold 3D coordinates
template<typename T>
struct TRd {
    T x, y, z;
};

// Templated function declarations
// Calculate the volume occupied by spheres using a grid-based approach
template<typename T>
T volume_of_spheres(const T* coords, const T* radii, size_t& n_spheres, T& grid_spacing);

// Get the maximum value from an array
template<typename T>
T get_max(const T* array, size_t& n);

// Calculate grid parameters: extent and origin
template<typename T>
void get_grid_params(
    const T* coords, const size_t& n, const T& cushion, const T& grid_spacing,
    TRi& extent, TRd<T>& origin
);

// Get the extent (min and max coordinates) from a list of 3D coordinates
template<typename T>
void get_extent(const T* coords, const size_t& n_coords, TRd<T>& min_coords, TRd<T>& max_coords);

#endif
