#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "msg.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double exact_u(double x, double Lx)
{
    return sin(M_PI*x/Lx);
}
static double rhs_f(double x, double Lx)
{
    return M_PI*M_PI/Lx/Lx*exact_u(x,Lx);
}


int main(int argc, char **argv)
{
    int Nx=128, i;
    double Lx=1.0;

    msg_init(&argc, &argv);

    msg_finalize();
    return 0;
}
