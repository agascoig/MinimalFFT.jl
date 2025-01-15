# MinimalFFT.jl

This is an implementation of the AbstractFFTs.jl interface in Julia.  All complex
tests in the AbstractFFTs testbench pass; the real tests are not fully supported
in this version.

This code has not been thoroughly tested and is mainly intended to better understand the AbstractFFTs package.

```
Test Summary:                  | Pass  Total  Time
correctness of fft, bfft, ifft | 1120   1120  3.4s

Test Summary:                     | Pass  Error  Total  Time
correctness of rfft, brfft, irfft |  509      1    510  3.9s

```
