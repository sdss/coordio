#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dimage.h"

/*
 * drefine.c
 *
 * Refine a set of identified peaks.
 *
 * Mike Blanton
 * 4/2022
 */

#define FREEVEC(a) {if((a)!=NULL) free((char *) (a)); (a)=NULL;}

static float *cimage=NULL;
static float *simage=NULL;
static int *peaks=NULL;

int drefine(float *image,
						int nx, 
						int ny,
						float *xrough, 
						float *yrough, 
						float *xrefined, 
						float *yrefined, 
						int ncen, 
						int cutout, /* assumed to be odd */
						float psf_sigma)
{
	int i,j,l,il,jl,ic,jc,ip,jp,oi,oj,npeaks,inearest;
	int ist,ind,jst,jnd,highest,ibrightest;
	float tmpxc,tmpyc,three[9],dnearest2,dnear2,brightest,bright;

	/* cutout */
	cimage=(float *) malloc(sizeof(float)*cutout*cutout);

	/* smoothed cutout */
	simage=(float *) malloc(sizeof(float)*cutout*cutout);

	/* list of peaks in a cutout */
	peaks=(int *) malloc(sizeof(int)*cutout*cutout);

	/* loop over centers */
	for(l=0;l<ncen;l++) {

		/* make cutout for peak */
		il = (int) floor(xrough[l]) - cutout/2;
		jl = (int) floor(yrough[l]) - cutout/2;
		for(ic=0;ic<cutout;ic++)
			for(jc=0;jc<cutout;jc++) {
				i=il+ic;
				j=jl+jc;
				if((i>=0) && (i<nx) && (j>=0) && (j<ny))
					cimage[ic + jc * cutout] = image[i + j * nx];
				else
					cimage[ic + jc * cutout] = 0.;
			} /* end for ic, jc */

		/* smooth cutout */
		dsmooth(cimage, cutout, cutout, psf_sigma, simage);

		/* find peaks, pick nearest original */
		/* -- Note this can be altered easily to pick brightest */
		npeaks=0;
		inearest=-1;
		dnearest2=1000000000.;
		ibrightest=-1;
		brightest=0.;
		for(jc=1;jc<cutout-1;jc++) {
			jst=jc-1;
			jnd=jc+1;
			for(ic=1;ic<cutout-1;ic++) {
				ist=ic-1;
				ind=ic+1;
				highest=1;
				for(ip=ist;ip<=ind;ip++)
					for(jp=jst;jp<=jnd;jp++)
						if(simage[ip+jp*cutout]>simage[ic+jc*cutout])
							highest=0;
				if(highest) {
					peaks[npeaks]=ic+jc*cutout;
					dnear2 = (ic - cutout/2) * (ic - cutout/2) +
						(jc - cutout/2) * (jc - cutout/2);
					if(dnear2 < dnearest2) {
						inearest = npeaks;
						dnearest2 = dnear2;
					} /* end if */
					bright = simage[peaks[npeaks]];
					if(bright > brightest) {
						ibrightest = npeaks;
						brightest = bright;
					} /* end if */
					npeaks++;
				} /* end if */
			} /* end for ic */
		} /* end for jc */
		ip = peaks[inearest]%cutout;
		jp = peaks[inearest]/cutout;

		/* now perform true refinement */
		if(ip>0 && ip<cutout-1 && jp>0 && jp<cutout-1) {
			for(oi=-1;oi<=1;oi++)
				for(oj=-1;oj<=1;oj++)
					three[oi+1+(oj+1)*3]= 
						simage[oi+ip+(oj+jp)*cutout];
			dcen3x3(three,&tmpxc,&tmpyc);
			xrefined[l]=tmpxc+(float)(il+ip-1);
			yrefined[l]=tmpyc+(float)(jl+jp-1);
		} /* end if */

	} /* end for l */
	
	FREEVEC(cimage);
	FREEVEC(simage);
	FREEVEC(peaks);

	return(1);
	
} /* end drefine */
