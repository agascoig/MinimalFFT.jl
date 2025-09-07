
# MinimalFFT.jl

This is an implementation of the AbstractFFTs.jl interface
in Julia, with some formal prime-factor indexing proofs in Lean.  This package easily allows for fixed-point and symbolic FFTs or different precision.  The results show just how good FFTW is, especially with small block sizes.
None of this package is currently hard coded in assembly language.

I am using AbstractFFTS.jl commit d64a878f3bda8b883a33f7bb05518ee8d8dfc707 Dec 14 2024.

## Tests

```
Test Summary:   | Pass  Total  Time
Project quality |   11     11  6.6s
Test Summary:                  | Pass  Total  Time
correctness of fft, bfft, ifft | 1120   1120  6.5s
Test Summary:                     | Pass  Total  Time
correctness of rfft, brfft, irfft |  540    540  4.3s
Test Summary: | Pass  Total  Time
rfft sizes    |    5      5  0.0s
Test Summary:   | Pass  Total  Time
Shift functions |   28     28  0.1s
Test Summary:   | Pass  Total  Time
FFT Frequencies |   71     71  0.3s
Test Summary: | Pass  Total  Time
normalization |    3      3  0.0s
Test Summary: | Pass  Total  Time
Default dims  |   18     18  0.2s
Test Summary:           | Pass  Total  Time
Complex float promotion |   15     15  0.0s
Test Summary:                    | Pass  Total  Time
Adjoint plan on single-precision |    3      3  0.7s
Test Summary:                                                  | Pass  Total  Time
Adjoint plan application when plan inverse is not a ScaledPlan |    3      3  0.1s
```

Note: ChainRules tests do not currently pass.  This is due to its attempt to fuzz the MinimalFFT plan.

## Performance

Of course, performance is very good with power of 2 block sizes
(a special case):

| FFT backend | Size (N) | Time |
|-----|------|------|
| FFTW.jl | 1 << 20 | 9.774 ms |
| MinimalFFT.jl | 1 << 20 | 1.332 ms |
| FFTW.jl | 1 << 22 | 53.972 ms |
| MinimalFFT.jl | 1 << 22 | 8.472 ms |

(Both FFTW and MinimalFFT are running on one thread only.  ComplexF64 is the data type.)

### Benchmark Procedure

```
using MinimalFFT, BenchmarkTools
N=1<<20;x=randn(ComplexF64,N);
P=plan_fft(x);
@btime (P * x);
```

These were obtained on an Apple Mac Mini M4 Pro processor.

Performance was significantly better (7x) with a Stockham-style radix-2 FFT than a
Cooley-Tukey algorithm, probably due to less memory conflicts (no need to do bit reversal
or load/store conflicts), prefetching, or better vectorization.

test/test17.jl shows planned performance comparisons with FFTW.  For small block sizes,
the performance is much worse than FFTW.  Larger block sizes are approximately twice
as slow as FFTW.

## Rader Algorithm

The Rader algorithm is slow, but is included for reference, as it is similar to the Winograd
decomposition for generating butterfly operations in
the generate subdirectory.

## Winograd FFT Butterfly Generation

This is not fully implemented (only shows proof of
reconstruction) due to difficulties with
Nemo, Symbolics, SymbolicUtils.  Namely, it is difficult to
use SymbolicUtils to factor the intermediate polynomials
as desired.

## Prime Factor Algorithm

A modern approach to the FFT is to use the Prime Factor Algorithm with CTA for each radix, which
is done here.  The code uses a non-standard scrambling/descrambling instead of Good-CRT
with lookup table.  See these documents for a summary:

[PFA 2 decomposition](https://agascoig.github.io/MinimalFFT.jl/pfa2.html)

[PFA 3 decomposition](https://agascoig.github.io/MinimalFFT.jl/pfa3.html)

