
using Revise
using Infiltrator

include("AbstractFFTsTestExt.jl")

#using RustFFT; test_complex_ffts(Array; test_inplace=true, test_adjoint=false);
#using FFTW
using MinimalFFT; # TestUtils.test_complex_ffts()

TestUtils.test_complex_ffts()
println("Completed test_complex_ffts()")
TestUtils.test_real_ffts()
println("Completed test_real_ffts()")
