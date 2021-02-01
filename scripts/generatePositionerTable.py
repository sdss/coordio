import numpy
import pandas
import os

from coordio.defaults import wokCoords

# metrology xy position in solid model
# mm in xy beta arm frame
modelMetXY = [14.314, 0]
# boss xy position in solid model
modelBossXY = [14.965, -0.376]
# apogee xy position in solid model
modelApXY = [14.965, 0.376]
alphaLen = 7.4 # mm

if __name__ == "__main__":
    positionerID = []
    robotailID = []
    wokID = []
    holeID = []
    apSpecID = []
    bossSpecID = []
    alphaArmLen = []
    metX = []
    metY = []
    apX = []
    apY = []
    bossX = []
    bossY = []

    pid = 0
    rid = 0
    for site in ["APO", "LCO"]:
        apid = 0
        boid = 0
        df = wokCoords[wokCoords["wokType"] == site]

        for ii, row in df.iterrows():
            if row.holeType not in ["Boss", "ApogeeBoss"]:
                continue

            wokID.append(site)
            holeID.append(row.holeID)

            # pIDstr = "FPU_%04d"%pid
            positionerID.append(pid)
            pid += 1

            # rIDstr = "RT_%04d"%rid
            robotailID.append(rid)
            rid += 1

            if row.holeType == "ApogeeBoss":
                apSpecID.append(apid)
                apid += 1
            else:
                apSpecID.append(None)

            bossSpecID.append(boid)
            boid += 1

            alphaArmLen.append(alphaLen)
            metX.append(modelMetXY[0])
            metY.append(modelMetXY[1])

            apX.append(modelApXY[0])
            apY.append(modelApXY[1])

            bossX.append(modelBossXY[0])
            bossY.append(modelBossXY[1])


    pDict = {}
    pDict["positionerID"] = positionerID
    pDict["robotailID"] = robotailID
    pDict["wokID"] = wokID
    pDict["holeID"] = holeID
    pDict["apSpecID"] = apSpecID
    pDict["bossSpecID"] = bossSpecID
    pDict["alphaArmLen"] = alphaArmLen
    pDict["metX"] = metX
    pDict["metY"] = metY
    pDict["apX"] = apX
    pDict["apY"] = apY
    pDict["bossX"] = bossX
    pDict["bossY"] = bossY

    positionerTable = pandas.DataFrame(pDict)

    filePath = os.path.join(
        os.path.dirname(__file__),
        "..", "coordio", "etc", "positionerTable.csv"
    )
    positionerTable.to_csv(filePath)



