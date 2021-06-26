import pytest
import numpy
numpy.random.seed(0)
import time
import pandas
# from coordio import Site, Wok, Observed, Field, FocalPlane, Tangent
from coordio.defaults import MICRONS_PER_MM, positionerTable, wokCoords, POSITIONER_HEIGHT
# import matplotlib.pyplot as plt
# import seaborn as sns

from coordio import conv


# get tangent coords at these locations
holeIDs = [
    "R0C14", "R0C1", "R+13C1", "R+13C14", "R0C27",
    "R-13C14", "R+13C7", "R-13C1"
]

tanCoordList = [
    [22, -4, 0],
    [-4, 180, 2],
    [60, 4, -1]
]


def test_wokAndTangent():

    for holeID in holeIDs:
        row = wokCoords[(wokCoords.holeID==holeID) & (wokCoords.wokType == "APO")]
        b = [round(float(row.x), 5), round(float(row.y), 5), round(float(row.z), 5)]
        iHat = [float(row.ix), float(row.iy), float(row.iz)]
        jHat = [float(row.jx), float(row.jy), float(row.jz)]
        kHat = [float(row.kx), float(row.ky), float(row.kz)]

        for tx,ty,tz in tanCoordList:
            scaleFac = numpy.random.uniform(0.9,1.1)
            dx = numpy.random.uniform(-0.01,0.01)
            dy = numpy.random.uniform(-0.01,0.01)
            dz = numpy.random.uniform(-0.01,0.01)
            dRot = numpy.random.uniform(-5,5)

            for addMe in [0, numpy.zeros(4)]:
                _tx = tx + addMe
                _ty = ty + addMe
                _tz = tz + addMe

                wx1,wy1,wz1 = conv.tangentToWok(_tx, _ty, _tz, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz, dRot=dRot)
                wx2,wy2,wz2 = conv._tangentToWok(_tx, _ty, _tz, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz, dRot=dRot)



                assert wx1 == pytest.approx(wx2)
                assert wy1 == pytest.approx(wy2)
                assert wz1 == pytest.approx(wz2)


                tx1,ty1,tz1 = conv.wokToTangent(wx1, wy1, wz1, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz, dRot=dRot)
                tx2,ty2,tz2 = conv._wokToTangent(wx1, wy1, wz1, b, iHat, jHat, kHat,
                    scaleFac=scaleFac, dx=dx, dy=dy, dz=dz, dRot=dRot)

                assert tx1 == pytest.approx(tx2)
                assert ty1 == pytest.approx(ty2)
                assert tz1 == pytest.approx(tz2)

                assert _tx == pytest.approx(tx2)
                assert _ty == pytest.approx(ty2)
                assert _tz == pytest.approx(tz2)

        # break

if __name__ == "__main__":
    test_wokAndTangent()
#     tx,ty,tz = numpy.zeros(1000)+22, numpy.zeros(1000)-4, numpy.zeros(1000)



#     import pdb; pdb.set_trace()

#     tstart = time.time()
#     _tx1, _ty1, _tz1 = conv.wokToTangent(wx,wy,wz,b,iHat,jHat,kHat)
#     print("carr", time.time()-tstart)
#     tstart = time.time()
#     _tx2, _ty2, _tz2 = conv._wokToTangent(wx,wy,wz,b,iHat,jHat,kHat)
#     print("narr", time.time()-tstart)


#     import pdb; pdb.set_trace()
#     break


# import pdb; pdb.set_trace()






