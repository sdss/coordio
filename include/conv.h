# pragma once
#include <cmath>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <Eigen/Geometry>

std::array<double, 3> wokToTangent(
    std::array<double, 3> & wokXYZ,
    std::array<double, 3> & basePos,
    std::array<double, 3> & iHat,
    std::array<double, 3> & jHat,
    std::array<double, 3> & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz,
    double dRot
);

std::vector<std::array<double,3>> wokToTangentArr(
    std::vector<std::array<double,3>> & wokXYZ,
    std::array<double, 3> & basePos,
    std::array<double, 3> & iHat,
    std::array<double, 3> & jHat,
    std::array<double, 3> & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz,
    double dRot
);
