/* Rectangular Dirichlet Poisson: discrete exactness + O(h^2) convergence. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "poisson.h"
#include "msg.h"
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif

static double exact_u(double x, double y, double Lx, double Ly)
{
    return sin(M_PI*x/Lx)*sin(2.0*M_PI*y/Ly);
}

/* -Laplacian of the above */
static double rhs_f(double x, double y, double Lx, double Ly)
{
    const double c = M_PI*M_PI/(Lx*Lx) + 4.0*M_PI*M_PI/(Ly*Ly);
    return c*exact_u(x,y,Lx,Ly);
}

int main(int argc, char **argv)
{
    int Nx=128, Ny=512, i,j;
    double Lx=2.0, Ly=1.0, e,emax;
    poisson_plan p;
    double *F,*U,*Uex,*Ud,*Fd;

    msg_init(&argc,&argv);
    if (msg_rank()==0) printf("running on %d rank(s)\n", num_ranks());

    if (argc>1) Nx = atoi(argv[1]);
    if (argc>2) Ny = atoi(argv[2]);
    if (argc>3) Lx = atof(argv[3]);
    if (argc>4) Ly = atof(argv[4]);
    int nx=Nx-1;
    int ny=Ny-1;

    if (poisson_plan_init(&p,nx,ny,Lx,Ly)) return 1;

    F  =(double*)malloc((size_t)nx*ny*sizeof(double));
    U  =(double*)malloc((size_t)nx*ny*sizeof(double));
    Uex=(double*)malloc((size_t)nx*ny*sizeof(double));
    Ud =(double*)malloc((size_t)nx*ny*sizeof(double));
    Fd =(double*)malloc((size_t)nx*ny*sizeof(double));
    if (!F||!U||!Uex||!Ud||!Fd) return 2;

    if (msg_rank()==0) {
        printf("domain  [0,%g] x [0,%g]\n",Lx,Ly);
        printf("nx ny   = %d %d      (Nx Ny = %d %d)\n",nx,ny,Nx,Ny);
        printf("hx hy   = %.6e %.6e\n",p.hx,p.hy);
    }

    /* ---- test 1: exact inversion of the DISCRETE operator ---------- */
    srand(999);
    for (i=0; i<nx*ny; ++i) Ud[i]=2.0*((double)rand()/(double)RAND_MAX)-1.0;
    poisson_residual_op(&p,Ud,Fd);
    poisson_solve(&p,Fd,U);
    poisson_solve(&p,Fd,U);
    emax=0.0;
    for (i=0; i<nx*ny; ++i) { e=fabs(U[i]-Ud[i]); if (e>emax) emax=e; }
    if (msg_rank()==0)
        printf("\ndiscrete solve  max|U - Uexact_h|      = %.6e   (expect ~roundoff)\n",emax);

    /* ---- test 2: continuous solution, expect O(h^2) ---------------- */
    for (j=0; j<ny; ++j) {
        double y=(double)(j+1)*p.hy;
        for (i=0; i<nx; ++i) {
            double x=(double)(i+1)*p.hx;
            F  [i+j*nx]=rhs_f  (x,y,Lx,Ly);
            Uex[i+j*nx]=exact_u(x,y,Lx,Ly);
        }
    }
    poisson_solve(&p,F,U);
    emax=0.0;
    for (i=0; i<nx*ny; ++i) { e=fabs(U[i]-Uex[i]); if (e>emax) emax=e; }
    double ep = emax*16;
    double em = emax/16;
    if (msg_rank()==0)
        printf("continuous      max|U - u(x,y)|        = %.6e  %.6e  %.6e  (expect O(h^2))\n",emax,em,ep);

    poisson_plan_free(&p);
    free(F); free(U); free(Uex); free(Ud); free(Fd);
    msg_finalize();
    return 0;
}
