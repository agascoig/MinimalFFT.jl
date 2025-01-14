
using MinimalFFT
using BenchmarkTools

function do_dfts(fn, Nlow, Nhigh, step)
    x = randn(ComplexF64,Nhigh)
    X = zeros(ComplexF64,Nhigh)
    x_time = zeros(Float64,Nhigh)

    for i=Nlow:step:Nhigh
        N = i
        println("N=",N)
        feedx = x[1:N]
        feedX = X[1:N]
        result = run(@benchmarkable $fn($feedx); samples=10, evals=1)
        x_time[N] = mean(result.times)
    end
    x_time
end

function direct_dft!(x)
    X=zeros(eltype(x),size(x))
    MinimalFFT.direct_dft(X, x)
    x.=X
end

y_time_direct = do_dfts(direct_dft!, 2, 128,1);
y_time_inner_fft = do_dfts(MinimalFFT.inner_fft!, 2, 128,1);
#fill_zeros_with_last_nonzero!(x_time_direct)

using Plots

function my_plot(y_time,clr)
    xseries = 1:length(y_time)
    xseries = xseries[y_time .!= 0]
    yfiltered = y_time[y_time .!= 0]
    scatter!(xseries, yfiltered, marker=:cross, color=clr)
end

my_plot(y_time_direct,:blue)
my_plot(y_time_inner_fft,:red)

