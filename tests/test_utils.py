from coordio.utils import radec2wokxy, wokxy2radec
import time
import matplotlib.pyplot as plt
import numpy
import coordio.fitData as fitData

from astropy.coordinates import SkyCoord
from astropy import units as u


# apo plate 15017
apo = {}
apo["utcJD"] = 2459249.6184
apo["alt"] = 54  # at the JD supplied...
apo["file"] = "plPlugMapP-15017.par"

# lco plate 12377
lco = {}
lco["utcJD"] = 2459249.8428
lco["alt"] = 45.18  # at the JD supplied
lco["file"] = "plPlugMapP-12377.par"


def parsePlugmap(plPlugFile):
    with open(plPlugFile) as f:
        rawLines = f.readlines()

    info = {}
    info["xFocal"] = []
    info["yFocal"] = []
    info["fiberType"] = []
    info["ra"] = []
    info["dec"] = []

    for line in rawLines:
        if line.startswith("ha "):
            info["ha"] = float(line.split()[1])
        if line.startswith("temp "):
            info["temp"] = float(line.split()[-1])
        if line.startswith("raCen "):
            info["raCen"] = float(line.split()[-1])
        if line.startswith("decCen "):
            info["decCen"] = float(line.split()[-1])
        if line.startswith("PLUGMAPOBJ "):
            if " QSO " in line:
                # assume this is a boss fiber
                split = line.split(" QSO ")[-1].split()
                info["xFocal"].append(float(split[0]))
                info["yFocal"].append(float(split[1]))

                split = line.split(" OBJECT ")[-1].split()
                info["ra"].append(float(split[0]))
                info["dec"].append(float(split[1]))

                info["fiberType"].append("Boss")
            elif " STAR_BHB " in line:
                split = line.split(" STAR_BHB ")[-1].split()
                info["xFocal"].append(float(split[0]))
                info["yFocal"].append(float(split[1]))

                split = line.split(" OBJECT ")[-1].split()
                info["ra"].append(float(split[0]))
                info["dec"].append(float(split[1]))

                info["fiberType"].append("Apogee")

    return info


def run_field(siteName, plot=False):
    if siteName == "LCO":
        dd = lco
    else:
        dd = apo
    plateData = parsePlugmap(dd["file"])

    xWok, yWok, fieldWarn, ha, pa = radec2wokxy(
        plateData["ra"], plateData["dec"], dd["utcJD"], plateData["fiberType"],
        plateData["raCen"], plateData["decCen"], 0, siteName, dd["utcJD"]
    )

    dHA = ha - plateData["ha"]

    # convert to hours
    dHA = 24/360.*dHA

    # convert to days
    dHA = dHA/24

    # update time of observation to be at designed hour angle
    timeObs = dd["utcJD"] - dHA

    xWok, yWok, fieldWarn, ha, pa = radec2wokxy(
        plateData["ra"], plateData["dec"], timeObs, plateData["fiberType"],
        plateData["raCen"], plateData["decCen"], 0, siteName, timeObs
    )

    print("obs ha", ha)
    print("design ha", plateData["ha"])
    xFocal = numpy.array(plateData["xFocal"])
    yFocal = numpy.array(plateData["yFocal"])

    # lco xy is backwards
    if siteName == "LCO":
        xFocal = xFocal*-1
        yFocal = yFocal*-1

    if plot:
        plt.figure(figsize=(8,8))
        plt.plot(xFocal, yFocal, 'x')
        plt.axis("equal")
        plt.title("focal")

        plt.figure(figsize=(8,8))
        plt.plot(xWok, yWok, 'x')
        plt.axis("equal")
        plt.title("wok")

    dx = xFocal - xWok
    dy = yFocal - yWok

    if plot:
        plt.figure()
        plt.hist(dx*1000)
        plt.xlabel("x err (micron)")

        plt.figure()
        plt.hist(dy*1000)
        plt.xlabel("y err (micron)")

        plt.figure()
        plt.hist(numpy.sqrt(dx**2+dy**2)*1000)
        plt.xlabel("r err (micron)")


    rmsErr = numpy.sqrt(numpy.sum(dx**2+dy**2) / len(dx))

    print("rms error (micron)", rmsErr*1000)

    if plot:
        plt.figure(figsize=(8,8))
        plt.title("Raw Residuals")
        plt.quiver(xWok,yWok,dx,dy, angles="xy")
        plt.axis("equal")


    # fit translation, rotation, scale
    fitTransRotScale = fitData.ModelFit(
        model=fitData.TransRotScaleModel(),
        measPos=numpy.array([xWok, yWok]).T,
        nomPos=numpy.array([xFocal, yFocal]).T,
        doRaise=True,
    )

    xyOff, rotAngle, scale = fitTransRotScale.model.getTransRotScale()
    print("xyOff (micron)", xyOff * 1000)
    print("rot (deg)", rotAngle)
    print("scale", scale)

    posErr = fitTransRotScale.getPosError()
    rmsErr = numpy.sqrt(numpy.sum(posErr**2) / len(posErr))
    print("fit rms error (micron)", rmsErr*1000)

    if siteName == "LCO":
        assert rmsErr*1000 < 1

    if plot:
        plt.figure()
        plt.hist(posErr[:,0]*1000)
        plt.xlabel("fit x err (micron)")

        plt.figure()
        plt.hist(posErr[:,1]*1000)
        plt.xlabel("fit y err (micron)")

        plt.figure()
        plt.hist(numpy.sqrt(posErr[:,0]**2+posErr[:,1]**2)*1000)
        plt.xlabel("fit r err (micron)")

        plt.figure(figsize=(8,8))
        plt.title("Fit Residuals")
        plt.quiver(xWok,yWok,posErr[:,0], posErr[:,1], angles="xy")
        plt.axis("equal")

    # run the reverse
    ra, dec, fieldWarn = wokxy2radec(
        xWok, yWok, plateData["fiberType"], plateData["raCen"],
        plateData["decCen"], 0, siteName, timeObs
    )

    sk1 = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    sk2 = SkyCoord(ra=plateData["ra"] * u.deg, dec=plateData["dec"] * u.deg)

    angSep = sk1.separation(sk2)
    asec = numpy.array(angSep)*3600
    assert numpy.max(asec) < 0.5  # less than 0.5 arcsecs round trip
    if plot:
        plt.figure()
        plt.hist(asec)
        plt.title("angular sep (arcsec)")
        plt.show()


def test_utils():
    run_field("APO")
    run_field("LCO")


if __name__ == "__main__":
    print("APO")
    print("-----------")
    run_field("APO", plot=True)
    print("\n\n")

    print("LCO")
    print("-----------")
    run_field("LCO", plot=True)
    # print("\n\n")
