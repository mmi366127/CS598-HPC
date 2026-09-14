
To run the code on the campus cluster:

./driver



# Rectangular Dirichlet Poisson via batched FST

A separate version of the square FST project, extended to a rectangular
domain with independent resolutions in x and y.

Build and run:

    make
    ./test_block_fst      [m] [n] [nrep]
    ./test_poisson        [nx] [ny] [Lx] [Ly]
    ./bench_transpose     [nrep]


## Storage convention

Everything is **column-major with the first index unit-stride**, i.e.
Fortran ordering:

    A(i,j)  ==  A[i + j*m],     i = 0..m-1  (unit stride)
                                j = 0..n-1  (stride m)


## `block_fst` on an m x n matrix

`block_fst_apply()` **contracts over the second ("n") index** and
**vectorizes over the first, unit-stride ("m") index**:

    A(i,k)  <-  sum_j  S_n(k,j) * A(i,j)

The m columns are the batch; n is the transform length. Every inner loop
in the fast path, the matmul reference, and the radix-4 butterfly runs
over `i` and is unit-stride in both operands.

`S_n` is the orthonormal DST-I matrix of order n,

    S_n(k,j) = sqrt(2/(n+1)) sin(pi*(j+1)*(k+1)/(n+1)),

with `S^T = S` and `S*S = I`, so the routine is its own inverse and no
separate backward transform is needed.

Restriction: `L = 2*(n+1) = 4^k`, so

    n = 31, 127, 511, 2047, ...

Three implementations, all with identical in-place semantics:

| routine | cost | notes |
|---|---|---|
| `block_fst_apply` | O(m n log n) | odd extension + batched radix-4 FFT |
| `block_fst_apply_matmul` | O(m n^2) | explicit S, unit-stride inner loop |
| `block_fst_apply_slow` | O(m n^2) | explicit S, naive triple loop |


## Rectangular transpose

    transpose_real(m, n, A, B)      A is m x n  ->  B is n x m
                                    B(j,i) = A(i,j)

In solver notation this maps `A(1:nx,1:ny)` to `B(1:ny,1:nx)`.

One of the two streams is necessarily non-unit-stride, so
`transpose_blocked` keeps a `TRANSPOSE_TS` square tile resident and moves
both streams in cache-line-sized runs. `TRANSPOSE_TS` defaults to 8 --
8 doubles is exactly one 64-byte cache line, which measured fastest
across every shape tested. `bench_transpose` compares it against the
naive version; expect roughly 1.2x-1.8x, largest for tall-thin shapes.


## Poisson solver

    -Laplacian u = f   on [0,Lx] x [0,Ly],   u = 0 on the boundary

with the standard 5-point stencil.

**Grid.** `Nx = nx+1` uniform intervals in x, so

    hx = Lx/Nx = Lx/(nx+1),    x_i = i*hx,  i = 1..nx
    hy = Ly/Ny = Ly/(ny+1),    y_j = j*hy,  j = 1..ny

`nx` and `ny` are the numbers of interior unknowns, arrays are `nx x ny`,
and each of `nx`, `ny` must satisfy `2*(n+1) = 4^k`. They need not match:
`127 x 511` and `2047 x 31` are both fine.

**Method.** Diagonalize in both directions:

    A  = F                                (nx x ny)
    A <- S_ny along y                     (contract 2nd index)
    B  = A^T                              (ny x nx)
    B <- S_nx along x                     (contract 2nd index)
    B(j,i) /= lam_x(i) + lam_y(j)
    B <- S_nx along x                     (S is its own inverse)
    A  = B^T                              (nx x ny)
    A <- S_ny along y
    U  = A

using the exact eigenvalues of the discrete operator,

    lam_x(i) = (4/hx^2) sin^2( pi*(i+1) / (2*(nx+1)) ).

Note the ordering: the transform always contracts the *second* index, so
the first pass hits y and the transpose is what exposes x.

**Verification.** `test_poisson` runs two checks.

1. *Discrete exactness.* Build `F = -Laplacian_h U` from a random `U`
   via `poisson_residual_op`, solve, and compare. Because the
   solver uses the eigenvalues of the same discrete operator, this
   returns `U` to roundoff (~1e-14), independent of truncation error.

2. *Continuous convergence.* With `u = sin(pi x/Lx) sin(2 pi y/Ly)` and
   the continuous `-Delta u` as data, the error is O(h^2). Measured
   ratios under 4x refinement:

       n =  31   2.734955e-03
       n = 127   1.706940e-04    ratio 16.02
       n = 511   1.066744e-05    ratio 16.00
