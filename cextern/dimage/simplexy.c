#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dimage.h"

/*
 * simplexy.c
 *
 * Get simple x and y values
 *
 * Mike Blanton
 * 1/2006 */

static float *invvar=NULL;
static float *mimage=NULL;
static float *simage=NULL;
static int *oimage=NULL;
static float *smooth=NULL;

#define PI 3.14159265358979

#define FREEVEC(a) {if((a)!=NULL) free((char *) (a)); (a)=NULL;}


int simplexy(float *image,
						 int nx, 
						 int ny,
						 float dpsf,  /* gaussian psf width; 1 is usually fine */
						 float plim,  /* significance to keep; 8 is usually fine */
						 float dlim,   /* closest two peaks can be; 1 is usually fine */
						 float saddle,  /* saddle difference (in sig); 3 is usually fine */
						 int maxper,   /* maximum number of peaks per object; 1000 */
						 int maxnpeaks,  /* maximum number of peaks total; 100000 */
						 float *sigma, 
						 float *x,  
						 float *y, 
						 float *flux, 
						 int *npeaks)
{
	float minpeak;
	int i,j;

	/* determine sigma */
	dsigma(image, nx, ny, 2, sigma);
	invvar=(float *) malloc(nx*ny*sizeof(float));
	for(j=0;j<ny;j++)
		for(i=0;i<nx;i++)
			invvar[i+j*nx]=1./((*sigma)*(*sigma));
	minpeak=(*sigma);
	
	/* median smooth */
	mimage=(float *) malloc(nx*ny*sizeof(float));
	simage=(float *) malloc(nx*ny*sizeof(float));
	dmedsmooth(image, invvar, nx, ny, 50, mimage);
	for(j=0;j<ny;j++)
		for(i=0;i<nx;i++)
			simage[i+j*nx]=image[i+j*nx]-mimage[i+j*nx];

	/* find objects */
	smooth=(float *) malloc(nx*ny*sizeof(float));
	oimage=(int *) malloc(nx*ny*sizeof(int));
	dobjects(simage, nx, ny, dpsf, plim, oimage);

	/* find all peaks within each object */
	dallpeaks(simage, nx, ny, oimage, x, y, npeaks, (*sigma), dlim, saddle, 
						maxper, maxnpeaks, minpeak);
	
	for(i=0;i<(*npeaks);i++) 
		flux[i]=simage[((int) x[i])+((int) y[i])*nx];

  FREEVEC(invvar);
  FREEVEC(mimage);
  FREEVEC(simage);
  FREEVEC(oimage);
  FREEVEC(smooth);
  
  return(1);
} /* end dmedsmooth */
