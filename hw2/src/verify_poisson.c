/* Verification deliverable: dump the full solution field + pointwise
 * error at one resolution (for a solution plot), and a max-error-vs-Nx
 * convergence sweep (for a convergence plot) -- see hw2.pdf "Data to
 * present: Verification that your code is correct."
 *
 * Uses the serial solver (poisson.c) as the correctness reference;
 * poisson_mpi.c has already been checked bit-for-bit against it at
 * P=1..16 in prior testing. */
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

static double rhs_f(double x, double y, double Lx, double Ly)
{
    const double c = M_PI*M_PI/(Lx*Lx) + 4.0*M_PI*M_PI/(Ly*Ly);
    return c*exact_u(x,y,Lx,Ly);
}

static double solve_case(int Nx, double Lx, double Ly, const char *dump_path)
{
    const int nx=Nx-1, ny=Nx-1;
    int i,j;
    poisson_plan p;
    double emax=0.0;

    if (poisson_plan_init(&p,nx,ny,Lx,Ly)) { fprintf(stderr,"plan_init failed\n"); exit(1); }

    double *F  =(double*)malloc((size_t)nx*ny*sizeof(double));
    double *U  =(double*)malloc((size_t)nx*ny*sizeof(double));
    double *Uex=(double*)malloc((size_t)nx*ny*sizeof(double));

    for (j=0; j<ny; ++j) {
        double y=(double)(j+1)*p.hy;
        for (i=0; i<nx; ++i) {
            double x=(double)(i+1)*p.hx;
            F  [i+j*nx]=rhs_f  (x,y,Lx,Ly);
            Uex[i+j*nx]=exact_u(x,y,Lx,Ly);
        }
    }

    poisson_solve(&p,F,U);   /* warm-up */
    poisson_solve(&p,F,U);

    for (i=0; i<nx*ny; ++i) {
        const double e = fabs(U[i]-Uex[i]);
        if (e>emax) emax=e;
    }

    if (dump_path) {
        FILE *fp = fopen(dump_path,"w");
        fprintf(fp,"i,j,x,y,U,Uex,err\n");
        for (j=0; j<ny; ++j) {
            double y=(double)(j+1)*p.hy;
            for (i=0; i<nx; ++i) {
                double x=(double)(i+1)*p.hx;
                const double u=U[i+j*nx], ue=Uex[i+j*nx];
                fprintf(fp,"%d,%d,%.10e,%.10e,%.10e,%.10e,%.10e\n",
                        i,j,x,y,u,ue,fabs(u-ue));
            }
        }
        fclose(fp);
        printf("wrote %s  (Nx=Ny=%d, %d x %d interior points)\n",dump_path,Nx,nx,ny);
    }

    poisson_plan_free(&p);
    free(F); free(U); free(Uex);
    return emax;
}

int main(int argc, char **argv)
{
    double Lx=2.0, Ly=1.0;
    msg_init(&argc,&argv);

    /* 1. Full field dump for a solution plot, at a middling resolution. */
    solve_case(128, Lx, Ly, "solution.csv");

    /* 2. Convergence sweep: max error vs Nx=Ny (valid sizes only,
     *    Nx-1 = 4^k-1: 31,127,511,2047). */
    {
        static const int Nxs[] = {32,128,512,2048};
        const int n = (int)(sizeof(Nxs)/sizeof(Nxs[0]));
        FILE *fp = fopen("convergence.csv","w");
        fprintf(fp,"Nx,nx,h,max_err\n");
        int k;
        for (k=0; k<n; ++k) {
            const int Nx = Nxs[k];
            const double h = Lx/(double)Nx;
            const double emax = solve_case(Nx, Lx, Ly, NULL);
            fprintf(fp,"%d,%d,%.10e,%.10e\n",Nx,Nx-1,h,emax);
            printf("Nx=Ny=%5d  h=%.6e  max_err=%.6e\n",Nx,h,emax);
            fflush(stdout);
        }
        fclose(fp);
    }

    msg_finalize();
    return 0;
}
