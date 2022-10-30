
#define SIGN(a,b) ((b)>0. ? fabs(a) : -fabs(a))

int simplexy(float *image, int nx, int ny, float dpsf, float plim,
						 float dlim, float saddle, int maxper, int maxnpeaks,
						 float *sigma, float *x, float *y, float *flux, int *npeaks);


int dsigma(float *image, int nx, int ny, int sp,float *sigma);

float dselip(unsigned long k, unsigned long n, float *arr);

int dmedsmooth(float *image, float *invvar, int nx, int ny, int box,
							 float *smooth);

int dobjects(float *image, int nx, int ny,
						 float dpsf, float plim, int *objects);

int dsmooth(float *image, int nx, int ny, float sigma, float *smooth);

int dfind(int *image, int nx, int ny, int *object);

int dallpeaks(float *image, int nx, int ny, int *objects, float *xcen,
							float *ycen, int *npeaks, float sigma, float dlim, float saddle,
							int maxper, int maxnpeaks, float minpeak);

int dpeaks(float *image, int nx, int ny, int *npeaks, int *xcen,
           int *ycen, float sigma, float dlim, float saddle, int maxnpeaks,
           int smooth, int checkpeaks, float minpeak, int abssaddle);

int dcen3x3(float *image, float *xcen, float *ycen);

int dfloodfill(int *image, int nx, int ny, int x, int y, int xst, int xnd,
							 int yst, int ynd, int nv);

int drefine(float *image, int nx, int ny, float *xrough, float *yrough,
						float *xrefined, float *yrefined, int ncen, int cutout,
						float smooth);
