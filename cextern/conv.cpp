#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
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
    double betaOffDeg,
    bool lefthand
){
    vec2 outArr;
    double alphaDeg, betaDeg;

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

    if (lefthand){
        alphaDeg = thetaTangent + xi - alphaOffDeg;
        betaDeg = 180 + gamma - thetaBAC - alphaOffDeg;
    }

    else {
        alphaDeg = thetaTangent - xi - alphaOffDeg;  // alpha angle
        betaDeg = 180.0 - gamma - thetaBAC - betaOffDeg;  // beta angle
    }

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
    double betaOffDeg,
    bool lefthand
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
                betaOffDeg,
                lefthand
            )
        );
    }
    return outArr;
}

double wrap2pi(double angle){
    // wrap an angle in radians
    // between 0 and 2 pi
    angle = fmod(angle, 2*M_PI);
    if (angle < 0.0){
        angle += 2*M_PI;
    }
    return angle;
}

double rad2deg(double radians){
    // convert from radians to degrees
    return radians * 180.0 / M_PI;
}

std::array<double, 5> tangentToPositioner2(
    vec2 xyTangent,
    vec2 xyBeta,
    double alphaLen,
    double alphaOffDeg,
    double betaOffDeg
){
    // new implementation
    // http://motion.pratt.duke.edu/RoboticSystems/InverseKinematics.html
    double alphaAngleRight, alphaAngleLeft, betaAngleRight, betaAngleLeft;
    double err;
    std::array<double, 5> output;

    double radDistT2 = xyTangent[0]*xyTangent[0] + xyTangent[1]*xyTangent[1];
    double radDistT = hypot(xyTangent[0], xyTangent[1]);
    double thetaT = atan2(xyTangent[1], xyTangent[0]);
    thetaT = wrap2pi(thetaT);

    double la2 = alphaLen*alphaLen;
    double la = alphaLen;
    double lb2 = xyBeta[0]*xyBeta[0] + xyBeta[1]*xyBeta[1];
    double lb = hypot(xyBeta[0], xyBeta[1]);
    if (radDistT >= la + lb){
        // outside donut
        betaAngleRight = 0;
        betaAngleLeft = 0;
        alphaAngleRight = thetaT;
        alphaAngleLeft = thetaT;
        err = radDistT - (la + lb);
    }
    else if (radDistT <= lb - la){
        // inside donut
        betaAngleRight = M_PI;
        betaAngleLeft = M_PI;
        alphaAngleRight = thetaT + M_PI;
        alphaAngleRight = wrap2pi(alphaAngleRight);
        alphaAngleLeft = alphaAngleRight;
        err = (lb-la) - radDistT;
    }
    else {
        // inside workspace, both left and right hand solutions exist
        double c_2 = (radDistT2 - la2 - lb2)/(2*la*lb);
        betaAngleRight = acos(c_2);
        betaAngleLeft = -1*betaAngleRight;

        alphaAngleRight = thetaT - atan2(
            lb*sin(betaAngleRight),
            la + lb*cos(betaAngleRight)
        );

        alphaAngleLeft = thetaT - atan2(
            lb*sin(betaAngleLeft),
            la + lb*cos(betaAngleLeft)
        );

        // wrap alpha beta angles to 0-360 degrees
        betaAngleRight = wrap2pi(betaAngleRight);
        betaAngleLeft = wrap2pi(betaAngleLeft);
        alphaAngleRight = wrap2pi(alphaAngleRight);
        alphaAngleLeft = wrap2pi(alphaAngleLeft);
        err = 0;
    }

    // account for offsets here
    // and slight angle off the robot arm (xyBeta)
    // fiber angular offset is the angle the fiber makes
    // with the "centerline" of the robot
    // alpha/beta should be reported with respect to the centerline
    // not the line connecting the beta axis and the fiber
    // eg fiberAngOff == 0 for a perfectly centered metrology fiber
    // no more wrapping!

    double fiberAngOff = rad2deg(atan2(xyBeta[1], xyBeta[0]));

    betaAngleRight = rad2deg(betaAngleRight) - betaOffDeg - fiberAngOff;
    betaAngleLeft = rad2deg(betaAngleLeft) - betaOffDeg - fiberAngOff;
    alphaAngleRight = rad2deg(alphaAngleRight) - alphaOffDeg;
    alphaAngleLeft = rad2deg(alphaAngleLeft) - alphaOffDeg;

    output[0] = alphaAngleRight;
    output[1] = betaAngleRight;
    output[2] = alphaAngleLeft;
    output[3] = betaAngleLeft;
    output[4] = err;

    return output;
}

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
){

    std::vector<vec2> outArr;
    vec2 xyTangent;
    double scaleFac = 1;

    int nCoords = wokXYZ.size();

    for (int ii = 0; ii < nCoords; ii++){
        vec3 tangentCoords = wokToTangent(
                wokXYZ[ii],
                basePos[ii],
                iHat[ii],
                jHat[ii],
                kHat[ii],
                elementHeight,
                scaleFac,
                dx[ii],
                dy[ii],
                dz[ii]
        );

        // note assume right handed coords only
        xyTangent[0] = tangentCoords[0];
        xyTangent[1] = tangentCoords[1];
        vec2 positionerCoords = tangentToPositioner(
            xyTangent,
            xyBeta[ii],
            alphaLen[ii],
            alphaOffDeg[ii],
            betaOffDeg[ii],
            false // not left handed
        );
        outArr.push_back(positionerCoords);
    }

    return outArr;

}

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
){

    std::vector<vec3> outArr;
    vec3 tangentCoords;
    double scaleFac = 1;

    int nCoords = alphaBetaDeg.size();

    for (int ii = 0; ii < nCoords; ii++){
        vec2 xyTangent = positionerToTangent(
            alphaBetaDeg[ii],
            xyBeta[ii],
            alphaLen[ii],
            alphaOffDeg[ii],
            betaOffDeg[ii]
        );

        tangentCoords[0] = xyTangent[0];
        tangentCoords[1] = xyTangent[1];
        tangentCoords[2] = 0; //elementHeight;

        vec3 xyzWok = tangentToWok(
            tangentCoords,
            basePos[ii],
            iHat[ii],
            jHat[ii],
            kHat[ii],
            elementHeight,
            scaleFac,
            dx[ii],
            dy[ii],
            dz[ii]
        );
        outArr.push_back(xyzWok);

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
    m.def("tangentToPositioner", &tangentToPositioner,
        "xyTangent"_a, "xyBeta"_a, "alphaLen"_a, "alphaOffDeg"_a,
        "betaOffDeg"_a, "leftHand"_a = false
    );
    m.def("tangentToPositionerArr", &tangentToPositionerArr,
        "xyTangent"_a, "xyBeta"_a, "alphaLen"_a, "alphaOffDeg"_a,
        "betaOffDeg"_a, "leftHand"_a = false
    );
    m.def("tangentToPositioner2", &tangentToPositioner2,
        "xyTangent"_a, "xyBeta"_a, "alphaLen"_a, "alphaOffDeg"_a,
        "betaOffDeg"_a
    );
    m.def("positionerToTangent", &positionerToTangent);
    m.def("positionerToTangentArr", &positionerToTangentArr);
    m.def("wokToPositionerArr", &wokToPositionerArr);
    m.def("positionerToWokArr", &positionerToWokArr);

}
