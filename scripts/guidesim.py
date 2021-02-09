import numpy

from coordio import Observed, Site, Field, FocalPlane, Wok, Tangent, Guide
import matplotlib.pyplot as plt

from scipy.optimize import minimize
import time

# note no wavelenghts are specified but coordio
# defaults the the guide wavelength, which is what
# we care about

gfas = ["GFA-S1", "GFA-S2", "GFA-S3", "GFA-S4", "GFA-S5", "GFA-S6"]
binX = 1
binY = 1
scale = 1
obsAngle = 0

nCoords = 30000

site = Site("APO")
site.set_time(2459249)

# choose where the telescope is pointing
# high alt where things get weird
altCen = 87
azCen = 160

# generate simulated stars in the field...a lot of them
altPts = numpy.random.uniform(-1.6, 1.6, size=nCoords) + altCen
azPts = numpy.random.uniform(
    -1.6/numpy.cos(numpy.radians(altCen)),
    1.6/numpy.cos(numpy.radians(altCen)),
    size=nCoords) + azCen

altAz = numpy.array([altPts, azPts]).T

obsCen = Observed([[altCen, azCen]], site=site)

obs = Observed(altAz, site=site)
field = Field(obs, field_center=obsCen)
focal = FocalPlane(field, site=site)
wok = Wok(focal, site=site, obsAngle=obsAngle)

gfaKeepInds = []
# goodInds holds indices of stars that fall on a GFA
for gfa in gfas:
    tc = Tangent(
        wok, site=site, holeID=gfa, scaleFactor=scale, obsAngle=obsAngle
    )
    gc = Guide(tc, binX=binX, binY=binY)
    args = numpy.argwhere(gc.guide_warn==False)
    if len(args) > 0:
        gfaKeepInds.append(args.squeeze())

# plot all the stars we just made,
# color label them by whether or not they are in the
# telescope FOV, and whether or not they land on a GFA
# note that the GFA's will be vignetted at APO...probably
plt.figure(figsize=(8,8))
plt.plot(azPts, altPts, '.k', alpha=0.3, label="field good")
plt.xlabel("az (deg)")
plt.ylabel("alt (deg)")

plt.plot(azPts[[focal.field_warn]], altPts[[focal.field_warn]], 'xk', alpha=0.3, label="field_warn")

for gfa, gfaInd in zip(gfas, gfaKeepInds):
    plt.plot(azPts[gfaInd], altPts[gfaInd], '.', label=gfa)

plt.legend(loc="center")
plt.savefig("field.png", dpi=150)
plt.close()


# keep only stars landing in a GFA, and orgainze them
# by which GFA they belong to
gfaDict = {}

for gfa, gfaInd in zip(gfas, gfaKeepInds):
    gfaDict[gfa] = {}
    gfaDict[gfa]["alt"] = altPts[gfaInd]
    gfaDict[gfa]["az"] = azPts[gfaInd]


# for each gfa predict the xy pixel for each star

def altaz2gfaxy(alt, az, gfaName, altCen=altCen,
                 azCen=azCen, obsAngle=obsAngle, scale=scale):

    altAz = numpy.array([alt, az]).T

    obsCen = Observed([[altCen, azCen]], site=site)

    obs = Observed(altAz, site=site)
    field = Field(obs, field_center=obsCen)
    focal = FocalPlane(field, site=site)
    wok = Wok(focal, site=site, obsAngle=obsAngle)
    tc = Tangent(
        wok, site=site, holeID=gfaName, scaleFactor=scale, obsAngle=obsAngle
    )
    gc = Guide(tc, binX=binX, binY=binY)

    # collect warnings for both telescope and GFA FOV
    warn = focal.field_warn | gc.guide_warn
    return gc[:,0], gc[:,1], warn


for gfa in gfas:
    alt = gfaDict[gfa]["alt"]
    az = gfaDict[gfa]["az"]
    x, y, warn = altaz2gfaxy(alt, az, gfa)
    gfaDict[gfa]["xPredict"] = x
    gfaDict[gfa]["yPredict"] = y

# next simulate a pointing error and simulate a psf measurement
dAlt = 50 / 3600. # 50 arcsec in degrees
dAz = 0.6 # degrees-20 / 3600. / numpy.cos(numpy.radians(altCen)) # -6 arcsecond in degrees
dRot = 1.5 #.5 #0.01 # deg

# this doesn't modify the plate scale
# of the telescope, it modifies the relative
# distance between GFA's (thermal expanson of the wok)
# appearing like  like
# a radial offset of the GFA's center, not
# a magnification of the telescope's image.
dScale = 0.99

# note: the other kind of scale (magnification) isn't yet
# handled, but it will be eventually, by adjusting the back focal
# distance of the wok


def plotOne(ax, d, xMeasure, yMeasure, warn, doLegend=False):
    # plot chip outline
    ax.plot([0,0], [0,2048], "--k")
    ax.plot([2048,2048], [0,2048], "--k")
    ax.plot([0,2048], [0,0], "--k")
    ax.plot([0,2048], [2048,2048], "--k")
    ax.plot([2048,2048], [0,2048], "--k")
    nGuide = len(xMeasure)

    gotWarn = False
    gotGood = False

    for ii in range(nGuide):
        xP = d["xPredict"][ii]
        yP = d["yPredict"][ii]
        xM = xMeasure[ii]
        yM = yMeasure[ii]
        _warn = warn[ii]
        if _warn:
            fillstyle = "none"
        else:
            fillstyle = "full"

        # plot the measured poin
        # xP = xM + dx (predicted is measured plus error)
        #  so arrows point from measured position to
        #  expected (desired)
        # arrow indicates where that psf should be if things were
        # aligned
        dx = xP - xM
        dy = yP - yM

        if _warn and not gotWarn:
            ax.plot(xM, yM, 'ok', fillstyle=fillstyle, label="cannot detect")
            gotWarn = True
        elif not _warn and not gotGood:
            ax.plot(xM, yM, 'ok', fillstyle=fillstyle, label="measured")
            gotGood = True
        else:
            ax.plot(xM, yM, 'ok', fillstyle=fillstyle)

        if ii == 0:
            ax.plot(xP, yP, 'xk', label="expected")
            ax.arrow(xM, yM, dx, dy, label="offset")
        else:
            ax.plot(xP, yP, 'xk')
            ax.arrow(xM, yM, dx, dy)
        if doLegend:
            ax.legend()


def plotPxOffsets():

    # cols are dAlt, dAz, dRot, dScale
    offsets = numpy.zeros((5,4))
    # scale should be 1 normally
    offsets[:,3] = 1
    offsets[0,0] = dAlt
    offsets[1,1] = dAz
    offsets[2,2] = dRot
    offsets[3,3] = dScale
    offsets[4,:] = numpy.array([dAlt, dAz, dRot, dScale])
    offsetTypes = ["dAlt", "dAz", "dRot", "dScale", "all"]

    for off, offType in zip(offsets, offsetTypes):
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs = axs.flatten()

        fig.suptitle("offset type: %s"%offType)

        doLegend = True
        for ax, gfa in zip(axs, gfas):
            # plot gfas in rows
            d = gfaDict[gfa]
            alt = d["alt"]
            az = d["az"]
            ax.set_title(gfa)
            ax.set_xlim([-500, 500+2048])
            ax.set_ylim([-500, 500+2048])
            # ax.axis("equal")
            nGuide = len(alt)

            _altCen = off[0] + altCen
            _azCen = off[1] + azCen
            _obsAngle = off[2] + obsAngle
            _scale = off[3]

            xMeasure, yMeasure, warn = altaz2gfaxy(
                alt, az, gfa, _altCen, _azCen, _obsAngle, _scale
            )

            plotOne(ax, d, xMeasure, yMeasure, warn, doLegend=doLegend)
            doLegend = False
        plt.savefig("%s.png"%offType, dpi=150)
        plt.close()

plotPxOffsets()
# plt.close("all")
# plt.show()

# try to solve for true pointing given expected positions
def minimizeMe(x):
    _altCen, _azCen, _obsAngle, _scale = x

    pxSqError = []

    for gfa in gfas:
        d = gfaDict[gfa]
        alt = d["alt"]
        az = d["az"]
        xMeasure, yMeasure, warn = altaz2gfaxy(
            alt, az, gfa, _altCen, _azCen, _obsAngle, _scale
            )
        dx = d["xPredict"] - xMeasure
        dy = d["yPredict"] - yMeasure

        sqErr = dx**2 + dy**2
        pxSqError.extend(list(sqErr))

    # return mean square error
    return numpy.mean(pxSqError)

xInit = numpy.array([altCen+dAlt, azCen+dAz, obsAngle+dRot, dScale])
print("Orig RMS Error (guide pixels)", numpy.sqrt(minimizeMe(xInit)))

t1 = time.time()
output = minimize(minimizeMe, xInit, method="nelder-mead")#, options={'xatol': 1e-2, 'ftol': 1e-2, 'disp': True})

print("nedler-mead took", time.time()-t1)
print("Fit RMS Error (guide pixels)", numpy.sqrt(minimizeMe(output.x)))


