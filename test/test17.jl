
using Primes, FFTW, Random, Printf, Dates

import Base.@inbounds
macro inbounds(ex)
    esc(ex) # Disabled bounds checking
end

using FFTW
using MinimalFFT

Random.seed!(6502)

fn_dict = Dict(2 => MinimalFFT.fftr2!,
    3 => MinimalFFT.fftr3!,
    4 => MinimalFFT.fftr4!,
    5 => MinimalFFT.fftr5!,
    7 => MinimalFFT.fftr7!,
    8 => MinimalFFT.fftr8!,
    9 => MinimalFFT.fftr9!,
    11 => MinimalFFT.fft_bluestein!,
    13 => MinimalFFT.fft_bluestein!
)

function power_of(b::Int, N::Int)::Int64
    count = 0
    if N <= 0
        return 0
    end
    while N % b == 0
        N ÷= b
        count += 1
        if N == 1
            return count
        end
    end
    0
end

const REPEAT_TEST = 3

function stockham(Y, X, inverse)
    N = length(X)
    f = factor(N)

    fn_dict[f[1][1]](Y, X, f[1][2], inverse)
end

using Infiltrator

function test_fft(name, N, pc::Ref{Int64}, fc::Ref{Int64}, fn1::Function, 
    P::Union{Nothing,MinimalFFT.MyPlan{T},MinimalFFT.ScaledPlan{T}},
    inverse::Bool, 
    args::Any...) where {T}

    Y_ref = zeros(ComplexF64, N)
    Y = zeros(ComplexF64, N)
    X_ref = randn(ComplexF64, N)
    X = copy(X_ref)

    rfftw_time = zeros(Float64, REPEAT_TEST)
    r_time = zeros(Float64, REPEAT_TEST)

    any_failed = false

    for i = 1:REPEAT_TEST
        P_ref = inverse ? FFTW.plan_ifft(X_ref) : FFTW.plan_fft(X_ref)

        result_ref = @timed P_ref * X_ref
        Y_ref = result_ref.value
        t_ref = result_ref.time

        t = 0.0
        
        if !(P === nothing)
            result = @timed P * X
            Y = result.value
            t = result.time
        else
            result = @timed fn1(Y, copy(X), args...)
            Y = result.value[1]
            t = result.time
        end

        if !(Y ≈ Y_ref)
            println("Failed for $name: N=$N error=", sum(abs.(Y - Y_ref)), " args=$args")
            any_failed = true
        end

        rfftw_time[i] = t_ref
        r_time[i] = t
    end

    if any_failed
        fc[] += 1
    else
        println("Passed for $name: N=$N= time=", @sprintf("%.8f", minimum(r_time)), " factor_ref=", @sprintf("%.3f", minimum(r_time) ./ minimum(rfftw_time)))
        pc[] += 1
    end
end

function driver(radix::Vector{Int64}, pc::Ref{Int64}, fc::Ref{Int64},
    parent_fn::Function, N_vals::Vector{Vector{Int64}}, inverse::Bool, name::String)

    for NF in N_vals
        bs = zeros(Int64, length(NF))
        es = zeros(Int64, length(NF))
        fns = Vector{Function}(undef, length(NF))
        i = 1
        for f in NF
            for j = 1:length(radix)
                r = radix[j]
                e = power_of(r, f)
                if e != 0
                    es[i] = e
                    bs[i] = r
                    fns[i] = fn_dict[r]
                    i += 1
                    break
                end
            end
        end
        if i != length(NF) + 1
            return
        end

        if length(NF) == 1
            test_fft(name, prod(NF), pc, fc, fns[1], nothing, inverse, es[1], false) # stockham
        else
            # prime_factor! or mixed_radix!
            test_fft(name, prod(NF), pc, fc, parent_fn, nothing, inverse, es..., NF..., fns..., false)
        end
    end
end

# test cases below

function do_tests()
    day = Dates.format(Dates.today(), "yyyy-mm-dd")
    t = Dates.format(Dates.now(), "HH:MM:SS")
    println("# $day time: $t\n\n")
    factor_1 = [[8], [27], [16], [125], [49], [64], [81], [256]]
    factor_2 = [[4, 25], [25, 4], [4, 49], [49, 4], [8, 9], [9, 8], [256, 25], [25, 256],
        [16, 5], [5, 16], [8, 7], [7, 8], [4, 25], [25, 4], [3, 49], [49, 3], [8, 9], [9, 8],
        [256, 25], [25, 256], [16, 5], [5, 16], [8, 7], [7, 8]]
    factor_3 = [[9, 5, 49], [9, 49, 5], [5, 9, 49], [5, 49, 9], [49, 9, 5], [49, 5, 9], [8, 25, 7],
        [8, 7, 25], [25, 8, 7], [25, 7, 8], [7, 8, 25], [7, 25, 8], [2, 3, 5], [2, 5, 3], [3, 2, 5],
        [3, 5, 2], [5, 2, 3], [5, 3, 2], [64, 3, 5], [64, 5, 3], [3, 64, 5], [3, 5, 64], [5, 64, 3],
        [5, 3, 64], [27, 625, 49], [27, 49, 625], [625, 27, 49], [625, 49, 27], [49, 27, 625], [49, 625, 27]]

    pc = Ref{Int64}(0)
    fc = Ref{Int64}(0)

    driver([2,3,5,7], pc, fc, stockham, factor_1, false, "stockham test 1")
    driver([2,9,5,7], pc, fc, stockham, factor_1, false, "stockham test 2")
    driver([4,3,5,7], pc, fc, stockham, factor_1, false, "stockham test 3")
    driver([4,9,5,7], pc, fc, stockham, factor_1, false, "stockham test 4")
    driver([8,3,5,7], pc, fc, stockham, factor_1, false, "stockham test 5")
    driver([8,9,5,7], pc, fc, stockham, factor_1, false, "stockham test 6")

    driver([2, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 1")
    driver([4, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 2")
    driver([8, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 3")
    driver([2, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 4")
    driver([4, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 5")
    driver([8, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_2, false, "prime factor 2 test 6")

    driver([2, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 1")
    driver([4, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 2")
    driver([8, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 3")
    driver([2, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 4")
    driver([4, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 5")
    driver([8, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_2, false, "mixed radix 2 test 6")

    driver([2, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 1")
    driver([4, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 2")
    driver([8, 3, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 3")
    driver([2, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 4")
    driver([4, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 5")
    driver([8, 9, 5, 7], pc, fc, MinimalFFT.prime_factor!, factor_3, false, "prime factor 3 test 6")

    driver([2, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 1")
    driver([4, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 2")
    driver([8, 3, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 3")
    driver([2, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 4")
    driver([4, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 5")
    driver([8, 9, 5, 7], pc, fc, MinimalFFT.mixed_radix!, factor_3, false, "mixed radix 3 test 6")

    planner_n = [100, 196, 72, 6400, 80, 56, 100, 147, 72, 2205,
        1400, 30, 960, 826875, 2 * 3 * 5 * 7 * 11 * 13]

    planner_n_inverse = planner_n

    for n in planner_n
        P = MinimalFFT.MyPlan{ComplexF64}(ComplexF64, (n,), 1, MinimalFFT.P_NONE)
        Pinv = MinimalFFT.inv(P)

        test_fft("planner_do_fft", n, pc, fc, MinimalFFT.execute_plan, P, false, 1) # planner_do_fft
        test_fft("planner_do_fft inverse", n, pc, fc, MinimalFFT.execute_plan, Pinv, true, 1) # planner_do_fft inverse
    end

    println("$(pc[]) tests passed.")
    println("$(fc[]) tests failed.")
end

today_str = Dates.format(Dates.today(), "yyyymmdd")
time_str = Dates.format(Dates.now(), "HHMMSS")
#do_tests()
@write_fn("../tmp/test17_$today_str" * "_" * "$time_str.txt", do_tests())