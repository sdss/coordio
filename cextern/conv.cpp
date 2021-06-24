#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "conv.h"

Eigen::MatrixXd wokToTangent(Eigen::MatrixXd & in)
    // std::vector<Eigen::Vector3d> & wokXYZ,
    // Eigen::Vector3d & basePos,
    // Eigen::Vector3d & iHat,
    // Eigen::Vector3d & jHat,
    // Eigen::Vector3d & kHat,
    // double elementHeight,
    // double scaleFac,
    // double dx,
    // double dy,
    // double dz,
    // double dRot,
{
    in.array() += 4;
    return in;
}

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(ccoordio, m) {
    m.def("wokToTangent", &wokToTangent);
}
