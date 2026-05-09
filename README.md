
# MinimalFFT.jl

This is an implementation of the AbstractFFTs.jl interface
in Julia.  This package easily allows for fixed-point and
symbolic FFTs (for verification of FFT implementations), or different precision.

The performance is not currently competitive with FFTW.

I am using AbstractFFTS.jl commit date Dec 14 2024 from GitHub.

## Performance

Of course, performance is very good with large power of 2 block sizes (a special case):

| FFT backend | Size (N) | Time |
|-----|------|------|
| FFTW.jl | 1 << 20 | 9.774 ms |
| MinimalFFT.jl | 1 << 20 | 10.787 ms |
| FFTW.jl | 1 << 22 | 53.972 ms |
| MinimalFFT.jl | 1 << 22 | 46.867 ms |

test/test17.jl shows performance comparisons with FFTW.  For small block sizes, the performance is much worse than FFTW.  Larger block sizes are currently approximately twice as slow as FFTW.

## Winograd FFT Butterfly Generation

This is not fully implemented (only shows can prove reconstruction) due to difficulties with Nemo, Symbolics, SymbolicUtils.  Namely, it is difficult to use SymbolicUtils to factor the intermediate polynomials as desired, at least as far as I know.

## Prime Factor Algorithm

A modern approach to the FFT is to use the Prime Factor Algorithm with CTA (here specifically Stockham) for each radix, which
is done here.

The code uses a non-standard scrambling/descrambling instead of Good-CRT with lookup table.  See https://github.com/agascoig/CPPMinimalFFT/docs
for details.

## Testing

An attempt is made to test this package in test/test17.jl.  This
takes about 6 minutes to run on my machine, and tries many
combinations of decompositions.  

### AbstractFFTs Tests

ChainRules tests do not currently pass.  This is due to its attempt to fuzz the MinimalFFT plan.  These are currently the only tests for multi-dimensional FFTs.

```
Test Summary:   | Pass  Total  Time
Project quality |   11     11  6.7s
Test Summary:                  | Pass  Total  Time
correctness of fft, bfft, ifft | 1120   1120  6.8s
Test Summary:                     | Pass  Total  Time
correctness of rfft, brfft, irfft |  540    540  4.5s
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

## Organization

| Function | |
|---------------------|-------------------------------------------|
| Lowest level functions | fftr2!, fftr3!, direct_dft!, fft_rader!, etc. |
| Mid level decomposition functions | prime_factor!, mixed_radix! |
| Indexer functions | do_1d, do_1d_r1 |
| Multi-dimensional FFT indexers | do_fft_planned, do_1d |
| Planning functions | execute_plan |
| High level functions | mul! |
