#ifndef CASES_H
#define CASES_H
/* Test problems on [0,L]^3, L = 1, shared by the serial and MPI drivers.
 *
 * sin   : f = sin(pi x/L) sin(pi y/L) sin(pi z/L), u = 0 on all faces,
 *         vertex-centered grid (h = L/(n+1)); exact u = f / (3 pi^2 / L^2).
 * face  : f = 0, u = 0 on x = 0, u = 1 on x = L, zero normal derivative on the four
 *         side faces.  Cell-centered grid (h = L/n).  Exact u = x/L.
 * cos   : as face, but u(L,y,z) = cos(pi y/L) cos(pi z/L).  Exact
 *         u = sinh(kx)/sinh(kL) cos(pi y/L) cos(pi z/L), k = sqrt(2) pi / L.
 * sincos: u(0,y,z) = sin(pi y/L) sin(pi z/L), u(L,y,z) = cos(pi y/L) cos(pi z/L).
 *         No closed form. */
#include <math.h>
#include <string.h>
#include "pcg.h"

#define CASE_L  1.0
#define CASE_PI 3.14159265358979323846

enum { CASE_SIN = 0, CASE_FACE, CASE_COS, CASE_SINCOS };

static inline int case_parse(const char *s)
{
    return !strcmp(s, "face") ? CASE_FACE : !strcmp(s, "cos") ? CASE_COS :
           !strcmp(s, "sincos") ? CASE_SINCOS : CASE_SIN;
}

static inline pcg_prob case_problem(int cs, int n)
{
    pcg_prob P = { n, 0.0, {0, 0, 0} };
    if (cs == CASE_SIN) P.h = CASE_L / (n + 1);
    else { P.h = CASE_L / n; P.g[0] = -1; P.g[1] = P.g[2] = 1; }
    return P;
}

/* coordinate of cell index i (global) */
static inline double case_x(int cs, const pcg_prob *P, int i)
{
    return (cs == CASE_SIN ? P->h : 0.5 * P->h) + i * P->h;
}

/* right-hand side of cell (i,j,k) (global indices), boundary data included */
static inline double case_rhs(int cs, const pcg_prob *P, int i, int j, int k)
{
    double L = CASE_L, PI = CASE_PI, h = P->h;
    double y = case_x(cs, P, j), z = case_x(cs, P, k);
    if (cs == CASE_SIN)
        return sin(PI * case_x(cs, P, i) / L) * sin(PI * y / L) * sin(PI * z / L);
    double b0 = cs == CASE_SINCOS ? sin(PI * y / L) * sin(PI * z / L) : 0.0;  /* u at x = 0 */
    double bL = cs == CASE_FACE ? 1.0 : cos(PI * y / L) * cos(PI * z / L);    /* u at x = L */
    return 2.0 / (h * h) * ((i == P->n - 1 ? bL : 0.0) + (i == 0 ? b0 : 0.0));
}

/* exact solution at cell (i,j,k), or NAN if none is known */
static inline double case_exact(int cs, const pcg_prob *P, int i, int j, int k)
{
    double L = CASE_L, PI = CASE_PI, kk = sqrt(2.0) * PI / L;
    double x = case_x(cs, P, i), y = case_x(cs, P, j), z = case_x(cs, P, k);
    switch (cs) {
    case CASE_SIN:  return case_rhs(cs, P, i, j, k) / (3.0 * PI * PI / (L * L));
    case CASE_FACE: return x / L;
    case CASE_COS:  return sinh(kk * x) / sinh(kk * L) * cos(PI * y / L) * cos(PI * z / L);
    default:        return NAN;
    }
}
#endif
