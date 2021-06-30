# pragma once
#include <cmath>
#include <vector>
#include <array>

typedef std::array<double, 3> vec3;
typedef std::array<double, 2> vec2;

vec3 wokToTangent(
    std::array<double, 3> & wokXYZ,
    std::array<double, 3> & basePos,
    std::array<double, 3> & iHat,
    std::array<double, 3> & jHat,
    std::array<double, 3> & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz
);

std::vector<vec3> wokToTangentArr(
    std::vector<std::array<double,3>> & wokXYZ,
    std::array<double, 3> & basePos,
    std::array<double, 3> & iHat,
    std::array<double, 3> & jHat,
    std::array<double, 3> & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz
);

vec2 positionerToTangent(
    vec2 alphaBetaDeg,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
);


vec2 tangentToPositioner(
    vec2 xyTangent,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
);

std::vector<vec2> positionerToTangentArr(
    std::vector<vec2> & alphaBetaDeg,
    std::vector<vec2> & xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
);

std::vector<vec2> tangentToPositionerArr(
    std::vector<vec2> & xyTangent,
    std::vector<vec2> & xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
);