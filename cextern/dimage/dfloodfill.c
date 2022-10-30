/* 
 * dfloodfill.c
 *
 * Based on Heckbert implementation described below.
 * Converted to ANSI C and defined simple pixelwrite and pixelread
 * 
 * Mike Blanton, June 2008
 */

/*
 * A Seed Fill Algorithm
 * by Paul Heckbert
 * from "Graphics Gems", Academic Press, 1990
 *
 * user provides pixelread() and pixelwrite() routines
 */

/*
 * fill.c : simple seed fill program
 * Calls pixelread() to read pixels, pixelwrite() to write pixels.
 *
 * Paul Heckbert	13 Sept 1982, 28 Jan 1987
 */

typedef struct {short y, xl, xr, dy;} Segment;
/*
 * Filled horizontal segment of scanline y for xl<=x<=xr.
 * Parent segment was on line y-dy.  dy=1 or -1
 */

#define MAX 10000		/* max depth of stack */

#define PUSH(Y, XL, XR, DY)	/* push new segment on stack */ \
    if (sp<stack+MAX && Y+(DY)>=yst && Y+(DY)<=ynd) \
    {sp->y = Y; sp->xl = XL; sp->xr = XR; sp->dy = DY; sp++;}

#define POP(Y, XL, XR, DY)	/* pop segment off stack */ \
    {sp--; Y = sp->y+(DY = sp->dy); XL = sp->xl; XR = sp->xr;}

#define pixelread(x, y) /* read a pixel out */	\
	image[y*nx+x]

#define pixelwrite(x, y, nv) /* read a pixel out */	\
	image[y*nx+x]=nv

/*
 * fill: set the pixel at (x,y) and all of its 4-connected neighbors
 * with the same pixel value to the new pixel value nv.
 * A 4-connected neighbor is a pixel above, below, left, or right of a pixel.
 */

int dfloodfill(int *image, 
							 int nx, 
							 int ny,
							 int x, 
							 int y, 
							 int xst, 
							 int xnd, 
							 int yst, 
							 int ynd, 
							 int nv)
{
	int l, x1, x2, dy;
	int ov;	/* old pixel value */
	Segment stack[MAX], *sp = stack;	/* stack of filled segments */
	
	ov = pixelread(x, y);		/* read pv at seed point */
	if (ov==nv || x<xst || x>xnd|| y<yst || y>ynd) return(1);
	PUSH(y, x, x, 1);			/* needed in some cases */
	PUSH(y+1, x, x, -1);		/* seed segment (popped 1st) */
	
	while (sp>stack) {
		/* pop segment off stack and fill a neighboring scan line */
		POP(y, x1, x2, dy);
		/*
		 * segment of scan line y-dy for x1<=x<=x2 was previously filled,
		 * now explore adjacent pixels in scan line y
		 */
		for (x=x1; x>=xst && pixelread(x, y)==ov; x--)
	    pixelwrite(x, y, nv);
		if (x>=x1) goto skip;
		l = x+1;
		if (l<x1) PUSH(y, l, x1-1, -dy);		/* leak on left? */
		x = x1+1;
		do {
	    for (; x<=xnd && pixelread(x, y)==ov; x++)
				pixelwrite(x, y, nv);
	    PUSH(y, l, x-1, dy);
	    if (x>x2+1) PUSH(y, x2+1, x-1, -dy);	/* leak on right? */
		skip:	    for (x++; x<=x2 && pixelread(x, y)!=ov; x++);
	    l = x;
		} while (x<=x2);
	}
	return(0);
}
