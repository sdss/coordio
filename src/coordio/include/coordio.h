# pragma once
#include <cmath>
#include <vector>
#include <array>

typedef std::array<double, 3> vec3;
typedef std::array<double, 2> vec2;

double dot3(vec3 & a, vec3 & b);

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
    vec3 & basePos,
    vec3 & iHat,
    vec3 & jHat,
    vec3 & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz
);

vec3 tangentToWok(
    vec3 & tangentXYZ,
    vec3 & basePos,
    vec3 & iHat,
    vec3 & jHat,
    vec3 & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz
);

std::vector<vec3> tangentToWokArr(
    std::vector<vec3> & tangentXYZ,
    vec3 & basePos,
    vec3 & iHat,
    vec3 & jHat,
    vec3 & kHat,
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
    double betaOffDeg,
    bool lefthand=false
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
    double betaOffDeg,
    bool leftand=false
);

std::array<double, 5> tangentToPositioner2(
    vec2 xyTangent,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
);

std::vector<vec3> positionerToWokArr(
    std::vector<std::array<double,2>> & alphaBetaDeg,
    std::vector<std::array<double,3>> & basePos,
    std::vector<std::array<double,3>> & iHat,
    std::vector<std::array<double,3>> & jHat,
    std::vector<std::array<double,3>> & kHat,
    double elementHeight,
    std::vector<double> & dx,
    std::vector<double> & dy,
    std::vector<double> & dz,
    std::vector<vec2> & xyBeta,
    std::vector<double> & alphaLen,
    std::vector<double> & alphaOffDeg,
    std::vector<double> & betaOffDeg
);

std::vector<vec2> wokToPositionerArr(
    std::vector<std::array<double,3>> & wokXYZ,
    std::vector<std::array<double,3>> & basePos,
    std::vector<std::array<double,3>> & iHat,
    std::vector<std::array<double,3>> & jHat,
    std::vector<std::array<double,3>> & kHat,
    double elementHeight,
    std::vector<double> & dx,
    std::vector<double> & dy,
    std::vector<double> & dz,
    std::vector<vec2> & xyBeta,
    std::vector<double> & alphaLen,
    std::vector<double> & alphaOffDeg,
    std::vector<double> & betaOffDeg
);
