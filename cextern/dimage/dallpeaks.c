#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dimage.h"

/*
 * dallpeaks.c
 *
 * Take image and list of objects, and produce list of all peaks (and
 * which object they are in). 
 *
 * BUGS: doesn't respect maxnpeaks
 *
 * Mike Blanton
 * 1/2006 */

#define PI 3.14159265358979

#define FREEVEC(a) {if((a)!=NULL) free((char *) (a)); (a)=NULL;}

static int *dobject=NULL;
static int *indx=NULL;
static float *oimage=NULL;
static float *simage=NULL;
static int *xc=NULL;
static int *yc=NULL;

int objects_compare(const void *first, const void *second)
{
  float v1,v2;
  v1=dobject[*((int *) first)];
  v2=dobject[*((int *) second)];
  if(v1>v2) return(1);
  if(v1<v2) return(-1);
  return(0);
}

int dallpeaks(float *image, 
							int nx, 
							int ny,
							int *objects, 
							float *xcen, 
							float *ycen, 
							int *npeaks, 
							float sigma, 
							float dlim, 
							float saddle, 
							int maxper,
							int maxnpeaks,
							float minpeak)
{
	int i,j,l,current,nobj,oi,oj,xmax,ymax,xmin,ymin,onx,ony,nc,lobj;
	int xcurr, ycurr, imore, maxsize;
	float tmpxc, tmpyc,three[9];
	
	maxsize=150;

  indx=(int *) malloc(sizeof(int)*nx*ny);
  dobject=(int *) malloc(sizeof(int)*(nx*ny+1));
	for(j=0;j<ny;j++)
		for(i=0;i<nx;i++)
			dobject[i+j*nx]=objects[i+j*nx];
  for(i=0;i<nx*ny;i++)
    indx[i]=i;
  qsort((void *) indx, nx*ny, sizeof(int), objects_compare);
	for(l=0;l<nx*ny && dobject[indx[l]]==-1; l++);
	nobj=0;
	(*npeaks)=0;
	oimage=(float *) malloc(sizeof(float)*nx*ny);
	simage=(float *) malloc(sizeof(float)*nx*ny);
	xc=(int *) malloc(sizeof(int)*maxper);
	yc=(int *) malloc(sizeof(int)*maxper);
	for(;l<nx*ny;) {
		current=dobject[indx[l]];

		/* get object limits */
		xmax=-1;
		xmin=nx+1;
		ymax=-1;
		ymin=ny+1;
		for(lobj=l;lobj<nx*ny && dobject[indx[lobj]]==current;lobj++) {
			xcurr=indx[lobj]%nx;
			ycurr=indx[lobj]/nx;
			if(xcurr<xmin) xmin=xcurr;
			if(xcurr>xmax) xmax=xcurr;
			if(ycurr<ymin) ymin=ycurr;
			if(ycurr>ymax) ymax=ycurr;
		}
		
		if(xmax-xmin>2 && ymax-ymin>2 && xmax-xmin<maxsize && ymax-ymin<maxsize) {
			/* make object cutout (if it is 3x3 or bigger) */
			onx=xmax-xmin+1;
			ony=ymax-ymin+1;
			for(oj=0;oj<ony;oj++) 
				for(oi=0;oi<onx;oi++) {
					oimage[oi+oj*onx]=0.;
					i=oi+xmin;
					j=oj+ymin;
					if(dobject[i+j*nx]==nobj) {
						oimage[oi+oj*onx]=image[i+j*nx];
					}
				}
			
			/* find peaks in cutout */
			dsmooth(oimage, onx, ony, 2, simage);
			dpeaks(simage, onx, ony, &nc, xc, yc, sigma, dlim, saddle, 
						 maxper, 0, 1, minpeak,0);
			imore=0;
			for(i=0;i<nc;i++) {
				if(xc[i]>0 && xc[i]<onx-1 && yc[i]>0 && yc[i]<ony-1) {
					for(oi=-1;oi<=1;oi++)
						for(oj=-1;oj<=1;oj++)
							three[oi+1+(oj+1)*3]= 
								simage[oi+xc[i]+(oj+yc[i])*onx];
					dcen3x3(three, &tmpxc,&tmpyc);
					xcen[imore+(*npeaks)]=tmpxc+(float)(xc[i]+xmin-1);
					ycen[imore+(*npeaks)]=tmpyc+(float)(yc[i]+ymin-1);
					imore++;
				}
			}
			(*npeaks)+=imore;
		}
		
		l=lobj;
		nobj++;
	}

	FREEVEC(indx);
	FREEVEC(dobject);
	FREEVEC(oimage);
	FREEVEC(simage);
	FREEVEC(xc);
	FREEVEC(yc);

	return(1);
	
} /* end dallpeaks */
