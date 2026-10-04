# Third-party notices

This package is an independent JAX implementation, distributed under the MIT
License in [LICENSE](LICENSE). Parts of it follow, translate or adapt the
software below, whose notices are reproduced here.

## L-BFGS-B 3.0

The solver follows the algorithm and control flow of L-BFGS-B 3.0 by Ciyou
Zhu, Richard Byrd and Jorge Nocedal, with the 3.0 revision described by José
Luis Morales and Jorge Nocedal: generalized Cauchy point, subspace
minimization, limited-memory updates and termination tests.
L-BFGS-B carries this condition for use, quoted from the SciPy 1.14.0
distribution (`scipy/optimize/lbfgsb_src/README`):

> This software is freely available, but we expect that all publications
> describing work using this software, or all commercial products using it,
> quote at least one of the references given below. This software is released
> under the BSD License.
>
> References
>   * R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound
>     Constrained Optimization, (1995), SIAM Journal on Scientific and
>     Statistical Computing, 16, 5, pp. 1190-1208.
>   * C. Zhu, R. H. Byrd and J. Nocedal. L-BFGS-B: Algorithm 778: L-BFGS-B,
>     FORTRAN routines for large scale bound constrained optimization (1997),
>     ACM Transactions on Mathematical Software, 23, 4, pp. 550 - 560.
>   * J.L. Morales and J. Nocedal. L-BFGS-B: Remark on Algorithm 778: L-BFGS-B,
>     FORTRAN routines for large scale bound constrained optimization (2011),
>     ACM Transactions on Mathematical Software, 38, 1.

## MINPACK-2 line search (`dcsrch`, `dcstep`)

`src/jax_lbfgsb/line_search.py` translates the More-Thuente line search
routines `dcsrch` and `dcstep` from MINPACK-2, as used by L-BFGS-B, into JAX:

> MINPACK-1 Project. June 1983. Argonne National Laboratory.
> Jorge J. More' and David J. Thuente.
>
> MINPACK-2 Project. November 1993. Argonne National Laboratory and University
> of Minnesota. Brett M. Averick, Richard G. Carter, and Jorge J. More'.

Reference: J. J. Moré and D. J. Thuente. Line search algorithms with
guaranteed sufficient decrease (1994), ACM Transactions on Mathematical
Software, 20, 3, pp. 286-307.

## SciPy

The solver's options, termination semantics and status messages mirror
`scipy.optimize.minimize(method="L-BFGS-B")`, whose implementation (including
its MINPACK-2 line search) is distributed under the following license:

```text
Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Optim.jl

`tests/test_optim_jl_cases.py` adapts test cases from the L-BFGS-B tests of
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl):

```text
Copyright (c) 2012: John Myles White, Tim Holy, and other contributors.
Copyright (c) 2016: Patrick Kofod Mogensen, John Myles White, Tim Holy,
                    and other contributors.
Copyright (c) 2017: Patrick Kofod Mogensen, Asbjørn Nilsen Riseth,
                    John Myles White, Tim Holy, and other contributors.

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
