#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * dcen3x3.c
 *
 * Find center of 3x3 image
 *
 * Mike Blanton
 * 1/2006 */

#define PI 3.14159265358979

#define FREEVEC(a) {if((a)!=NULL) free((char *) (a)); (a)=NULL;}

int dcen3(float f0, float f1, float f2, float *xcen) 
{
	float s, d, aa, sod, kk;
	
	kk=1.3333;
	s=0.5*(f2-f0);
	d=2.*f1-(f0+f2);
	
	if(d<=1.e-10*f0) {
		if(f0>f1 && f0>f2) (*xcen)=0.;
		if(f2>f1 && f2>f0) (*xcen)=2.;
		(*xcen)=1.;
		return(1);
	}

	aa=f1+0.5*s*s/d;
	sod=s/d;
	(*xcen)=sod*(1.+kk*(0.25*d/aa)*(1.-4.*sod*sod))+1.;

	return(1);
}

int dcen3x3(float *image, float *xcen, float *ycen)
{
	float mx0, mx1, mx2;
	float my0, my1, my2;
	float bx, by, mx ,my;
	
	dcen3(image[0+3*0], image[1+3*0], image[2+3*0], &mx0);
	dcen3(image[0+3*1], image[1+3*1], image[2+3*1], &mx1);
	dcen3(image[0+3*2], image[1+3*2], image[2+3*2], &mx2);

	dcen3(image[0+3*0], image[0+3*1], image[0+3*2], &my0);
	dcen3(image[1+3*0], image[1+3*1], image[1+3*2], &my1);
	dcen3(image[2+3*0], image[2+3*1], image[2+3*2], &my2);

	/* x = (y-1) mx + bx */
	bx=(mx0+mx1+mx2)/3.;
	mx=(mx2-mx0)/2.;

	/* y = (x-1) my + by */
	by=(my0+my1+my2)/3.;
	my=(my2-my0)/2.;

	(*xcen)=(mx*(by-my-1.)+bx)/(1.+mx*my);
	(*ycen)=((*xcen)-1.)*my+by;
	
	return(1);
} /* end dcen3x3 */
