#ifndef BLOCK_FFT_H
#define BLOCK_FFT_H
#include "cpx.h"
typedef struct {
    int n, q, radix, stages;
    int *rev;
    cpx *W1, *W2, *W3;
} block_fft_plan;
int  block_fft_plan_init(block_fft_plan *p, int n, int q, int radix);
void block_fft_plan_free(block_fft_plan *p);
void block_fft_forward(block_fft_plan *p, cpx * restrict x);
#endif
