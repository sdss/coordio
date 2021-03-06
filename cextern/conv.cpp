#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "coordio.h"

// #include <pybind11/eigen.h>


double dot3(vec3 & a, vec3 & b){
    // might wanna consider numerical stability here?
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


vec3 rigidTransform(
    vec3 & inArr, vec3 & iHat, vec3 & jHat, vec3 & kHat, bool invert = false
){
    vec3 outArr, xDir, yDir, zDir;

    if (invert) {
        // transpose the ijk's
        xDir[0] = iHat[0];
        xDir[1] = jHat[0];
        xDir[2] = kHat[0];

        yDir[0] = iHat[1];
        yDir[1] = jHat[1];
        yDir[2] = kHat[1];

        zDir[0] = iHat[2];
        zDir[1] = jHat[2];
        zDir[2] = kHat[2];
    } else {
        xDir = iHat;
        yDir = jHat;
        zDir = kHat;
    }

    outArr[0] = dot3(xDir, inArr);
    outArr[1] = dot3(yDir, inArr);
    outArr[2] = dot3(zDir, inArr);
    return outArr;
}

vec3 transScaleXY(vec3 & inArr, vec3 & basePos, double scaleFac, bool invert = false){
    // translate and scale inArr by basePos and scaleFac
    vec3 outArr;
    double xscaled = basePos[0];
    double yscaled = basePos[1];
    bool atOrigin = basePos[0] == 0.0 and basePos[1] == 0.0;

    if ((scaleFac != 1) & (!atOrigin)) {
        // std::cout << "applying scale" << std::endl;
        auto r = hypot(basePos[0], basePos[1])*scaleFac;
        auto theta = atan2(basePos[1], basePos[0]);
        xscaled = r * cos(theta);
        yscaled = r * sin(theta);
    }

    if (invert){
        outArr[0] = inArr[0] + xscaled;
        outArr[1] = inArr[1] + yscaled;
        outArr[2] = inArr[2] + basePos[2];
    } else {
        outArr[0] = inArr[0] - xscaled;
        outArr[1] = inArr[1] - yscaled;
        outArr[2] = inArr[2] - basePos[2];
    }

    return outArr;
}

vec3 wokToTangent(
    vec3 & wokXYZ,
    vec3 & basePos,
    vec3 & iHat,
    vec3 & jHat,
    vec3 & kHat,
    double elementHeight,
    double scaleFac,
    double dx,
    double dy,
    double dz
){

    vec3 tangentXYZ = wokXYZ;


    tangentXYZ = transScaleXY(tangentXYZ, basePos, scaleFac);

    // rotate normal to wok surface at point b
    tangentXYZ = rigidTransform(tangentXYZ, iHat, jHat, kHat);

    // offset xy plane to focal surface
    tangentXYZ[2] -= elementHeight;

    // apply rotational calibrations
    // if (dRot != 0){
    //     // convert dRot to radians
    //     tangentXYZ = rotZ(tangentXYZ, dRot);
    // }

    // apply offset calibrations
    if (dx != 0){
        tangentXYZ[0] -= dx;
    }
    if (dy != 0){
        tangentXYZ[1] -= dy;
    }
    if (dz != 0){
        tangentXYZ[2] -= dz;
    }
    return tangentXYZ;
}

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
    // double dRot
){

    vec3 wokXYZ = tangentXYZ;

    // apply offset calibrations
    if (dx != 0){
        wokXYZ[0] += dx;
    }
    if (dy != 0){
        wokXYZ[1] += dy;
    }
    if (dz != 0){
        wokXYZ[2] += dz;
    }

    // apply rotational calibrations
    // if (dRot != 0){
    //     // convert dRot to radians
    //     wokXYZ = rotZ(wokXYZ, dRot, true);
    // }

    wokXYZ[2] += elementHeight;

    wokXYZ = rigidTransform(wokXYZ, iHat, jHat, kHat, true);

    wokXYZ = transScaleXY(wokXYZ, basePos, scaleFac, true);

    return wokXYZ;
}


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
){
    std::vector<std::array<double,3>> outArr;
    int nCoords = wokXYZ.size();

    for (int ii = 0; ii < nCoords; ii++){
        outArr.push_back(
            wokToTangent(
                wokXYZ[ii],
                basePos,
                iHat,
                jHat,
                kHat,
                elementHeight,
                scaleFac,
                dx,
                dy,
                dz
            )
        );
    }

    return outArr;
}

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
){
    std::vector<vec3> outArr;
    int nCoords = tangentXYZ.size();

    for (int ii = 0; ii < nCoords; ii++){
        outArr.push_back(
            tangentToWok(
                tangentXYZ[ii],
                basePos,
                iHat,
                jHat,
                kHat,
                elementHeight,
                scaleFac,
                dx,
                dy,
                dz
            )
        );
    }

    return outArr;
}

vec2 positionerToTangent(
    vec2 alphaBetaDeg,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
){

    vec2 outArr;

    auto betaOffRad = betaOffDeg * M_PI / 180.0;
    auto alphaOffRad = alphaOffDeg * M_PI / 180.0;
    auto alphaRad = alphaBetaDeg[0] * M_PI / 180.0;
    auto betaRad = alphaBetaDeg[1] * M_PI / 180.0;

    auto thetaBAC = atan2(xyBeta[1], xyBeta[0]);  // radians!
    auto rBAC = hypot(xyBeta[0], xyBeta[1]);
    auto cosAlpha = cos(alphaRad + alphaOffRad);
    auto sinAlpha = sin(alphaRad + alphaOffRad);
    auto cosAlphaBeta = cos(alphaRad + betaRad + thetaBAC + betaOffRad + alphaOffRad);
    auto sinAlphaBeta = sin(alphaRad + betaRad + thetaBAC + betaOffRad + alphaOffRad);

    outArr[0] = alphaLen * cosAlpha + rBAC * cosAlphaBeta;
    outArr[1] = alphaLen * sinAlpha + rBAC * sinAlphaBeta;
    return outArr;

}

vec2 tangentToPositioner(
    vec2 xyTangent,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
){
    vec2 outArr;

    auto thetaTangent = atan2(xyTangent[1], xyTangent[0]);

    // polar coords jive better for this calculation
    auto rTangentSq = xyTangent[0]*xyTangent[0] + xyTangent[1]*xyTangent[1];
    auto rTangent = hypot(xyTangent[0], xyTangent[1]);

    // convert xy Beta to radial coords
    // the origin of the beta coord system is the
    // beta axis of rotation
    auto thetaBAC = atan2(xyBeta[1], xyBeta[0]); // radians!
    auto rBacSq = xyBeta[0]*xyBeta[0] + xyBeta[1]*xyBeta[1];
    auto rBac = hypot(xyBeta[0], xyBeta[1]);
    auto la2 = alphaLen*alphaLen;

    auto gamma = acos(
        (la2 + rBacSq - rTangentSq) / (2 * alphaLen * rBac)
    );
    auto xi = acos(
        (la2 + rTangentSq - rBacSq) / (2 * alphaLen * rTangent)
    );

    thetaTangent = thetaTangent * 180.0 / M_PI;
    thetaBAC = thetaBAC * 180.0 / M_PI;
    gamma = gamma * 180.0 / M_PI;
    xi = xi * 180.0 / M_PI;

    auto alphaDeg = thetaTangent - xi - alphaOffDeg;  // alpha angle
    auto betaDeg = 180.0 - gamma - thetaBAC - betaOffDeg;  // beta angle

    alphaDeg = fmod(alphaDeg,360);
    if (alphaDeg < 0.0)
        alphaDeg += 360.0;

    outArr[0] = alphaDeg;
    outArr[1] = betaDeg;

    return outArr;
}

std::vector<vec2> positionerToTangentArr(
    std::vector<vec2> & alphaBetaDeg,
    std::vector<vec2> & xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
){
    std::vector<vec2> outArr;
    int nCoords = alphaBetaDeg.size();

    for (int ii = 0; ii < nCoords; ii++){
        outArr.push_back(
            positionerToTangent(
                alphaBetaDeg[ii],
                xyBeta[ii],
                alphaLen,
                alphaOffDeg,
                betaOffDeg
            )
        );
    }
    return outArr;
}

std::vector<vec2> tangentToPositionerArr(
    std::vector<vec2> & xyTangent,
    std::vector<vec2> & xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
){
    std::vector<vec2> outArr;
    int nCoords = xyTangent.size();

    for (int ii = 0; ii < nCoords; ii++){
        outArr.push_back(
            tangentToPositioner(
                xyTangent[ii],
                xyBeta[ii],
                alphaLen,
                alphaOffDeg,
                betaOffDeg
            )
        );
    }
    return outArr;
}



namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(libcoordio, m) {
    m.def("wokToTangent", &wokToTangent);
    m.def("wokToTangentArr", &wokToTangentArr);
    m.def("tangentToWok", &tangentToWok);
    m.def("tangentToWokArr", &tangentToWokArr);
    m.def("tangentToPositioner", &tangentToPositioner);
    m.def("tangentToPositionerArr", &tangentToPositionerArr);
    m.def("positionerToTangent", &positionerToTangent);
    m.def("positionerToTangentArr", &positionerToTangentArr);
}
