
using Primes, FFTW, Random, Printf, Dates

using FFTW
using MinimalFFT
using BenchmarkTools
using Infiltrator
Random.seed!(6502)
BenchmarkTools.DEFAULT_PARAMETERS.samples = 10

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

function stockham(Y, X, fn1, exponent, inverse)
    N = length(X)
    Y, X = fn1(Y, X, N, exponent, 1, 1, inverse)
    Y, X
end

function test_fft(name::String, bm::Bool, inverse::Bool, N::Int64, pc::Ref{Int64}, fc::Ref{Int64}, fn1::Function,
    P::Union{Nothing,MinimalFFT.MinimalPlan{T},MinimalFFT.ScaledPlan{T}},
    args::Any...) where {T}

    Y_ref = zeros(ComplexF64, N)
    Y = zeros(ComplexF64, N)
    X_ref = randn(ComplexF64, N)
    X = copy(X_ref)
    copy_X = copy(X)

    P_ref = inverse ? FFTW.inv(FFTW.plan_fft(X_ref)) : FFTW.plan_fft(X_ref)

    if P_ref isa AbstractFFTs.ScaledPlan
        @assert P_ref.p isa FFTW.cFFTWPlan "Reference plan must be FFTW.cFFTWPlan"
    else
        @assert P_ref isa FFTW.cFFTWPlan "Reference plan must be FFTW.cFFTWPlan"
    end

    if bm
        result_ref = @btimed $P_ref * copy($X_ref)
        Y_ref = result_ref.value
        t_ref = result_ref.time
        t = 1.0

        if P !== nothing
            result = @btimed $P * copy($X)
            Y = result.value
            t = result.time
            @assert X ≈ X_ref "MinimalFFT modified input." # TBD: allow for inplace
        else
            result = @btimed $fn1($Y, copy($X), $args...)
            Y = result.value[1]
            t = result.time
            if inverse
                Y .*= (1.0 / N)
            end
        end
    else
        Y_ref = P_ref * copy(X_ref)
        t_ref = 1.0
        t = 1.0

        if P !== nothing
            Y = P * X
            @assert X ≈ X_ref "MinimalFFT modified input." # TBD: allow for inplace
        else
            Y, X = fn1(Y, X, args...)
            if inverse
                Y .*= (1.0 / N)
            end
        end
    end

    if !(Y ≈ Y_ref)
        println("Failed for $name: N=$N error=", sum(abs.(Y - Y_ref)), " args=$args")
        fc[] += 1
    else
        if bm
            println("Passed for $name: N=$N time=", @sprintf("%.2e",t) , " args=$args", " factor_ref=", @sprintf("%.3f", t / t_ref))
        else
            println("Passed for $name: N=$N args=$args")
        end
        pc[] += 1
    end
end

function driver(radix::Vector{Int64}, bm::Bool, pc::Ref{Int64}, fc::Ref{Int64}, inverse::Bool,
    parent_fn::Function, N_vals::Vector{Vector{Int64}}, name::String)

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
            continue
        end

        if length(NF) == 1
            test_fft(name, bm, inverse, prod(NF), pc, fc, parent_fn, nothing, fns[1], es[1], inverse) # stockham
        else
            # prime_factor! or mixed_radix!
            test_fft(name, bm, inverse, prod(NF), pc, fc, parent_fn, nothing, es..., NF..., fns..., 1, 1, inverse)
        end
    end
end

# test cases below

function do_tests()
    day = Dates.format(Dates.today(), "yyyy-mm-dd")
    t = Dates.format(Dates.now(), "HH:MM:SS")
    println("# $day time: $t\n\n")
    factor_1 = [[8], [25], [27], [16], [125], [49], [64], [81], [9*9*9], [256]]
    factor_2 = [[4, 25], [25, 4], [4, 49], [8, 9], [256, 25], [25, 256],
        [16, 5], [8, 7], [11, 8], [25, 4], [49, 3], [9, 8],
        [25, 256], [16, 5], [8, 7], [25, 4], [1,256]]
    factor_3 = [[9, 5, 49], [9, 49, 5], [5, 9, 49], [49, 5, 9],
        [8, 7, 25], [7, 25, 8], [2, 3, 5], [2, 5, 3], [3, 2, 5],
        [3, 5, 2], [64, 3, 5], [3, 5, 64], [5, 64, 3], [1, 1, 64],
        [27, 625, 49], [625, 27, 49], [49, 27, 625], [49, 625, 27]]

    pc = Ref{Int64}(0)
    fc = Ref{Int64}(0)
 
    driver([2, 3, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 0")
    driver([2, 3, 5, 7], true, pc, fc, false, stockham, factor_1, "stockham test 1")
    driver([2, 9, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 2")
    driver([4, 3, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 3")
    driver([4, 9, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 4")
    driver([8, 3, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 5")
    driver([8, 9, 5, 7], false, pc, fc, false, stockham, factor_1, "stockham test 6")

    driver([2, 3, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 0")
    driver([2, 3, 5, 7], true, pc, fc, true, stockham, factor_1, "stockham inverse test 1")
    driver([2, 9, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 2")
    driver([4, 3, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 3")
    driver([4, 9, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 4")
    driver([8, 3, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 5")
    driver([8, 9, 5, 7], false, pc, fc, true, stockham, factor_1, "stockham inverse test 6")

    driver([2, 3, 5, 7], false, pc, fc, true, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 0")
    driver([2, 3, 5, 7], true, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 1")
    driver([4, 3, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 2")
    driver([8, 3, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 3")
    driver([2, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 4")
    driver([4, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 5")
    driver([8, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 6")

    driver([2, 3, 5, 7], false, pc, fc, true, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 0")
    driver([2, 3, 5, 7], true, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 1")
    driver([4, 3, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 2")
    driver([8, 3, 5, 7], false, pc, fc, false ,MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 3")
    driver([2, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 4")
    driver([4, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 5")
    driver([8, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 6")

    driver([2, 3, 5, 7], false, pc, fc, true, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 0")
    driver([2, 3, 5, 7], true, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 1")
    driver([4, 3, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 2")
    driver([8, 3, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 3")
    driver([2, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 4")
    driver([4, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 5")
    driver([8, 9, 5, 7], false, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 6")

    driver([2, 3, 5, 7], false, pc, fc, true, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 0")
    driver([2, 3, 5, 7], true, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 1")
    driver([4, 3, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 2")
    driver([8, 3, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 3")
    driver([2, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 4")
    driver([4, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 5")
    driver([8, 9, 5, 7], false, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 6")

    planner_n = [100, 196, 72, 6400, 80, 56, 100, 147, 72, 2205,
        1400, 30, 960, 826875, 2 * 3 * 5 * 7 * 11 * 13]

    planner_n_inverse = planner_n

    for n in planner_n
        P = MinimalFFT.MinimalPlan{ComplexF64}(ComplexF64, (n,), 1, MinimalFFT.P_NONE)
        Pinv = MinimalFFT.inv(P)

        test_fft("execute_plan", true, false, n, pc, fc, MinimalFFT.execute_plan, P, 1, 1, 1) # planner_do_fft
        test_fft("execute_plan inverse", true, true, n, pc, fc, MinimalFFT.execute_plan, Pinv, 1, 1, 1) # planner_do_fft inverse
    end

    println("$(pc[]) tests passed.")
    println("$(fc[]) tests failed.")
end

today_str = Dates.format(Dates.today(), "yyyymmdd")
time_str = Dates.format(Dates.now(), "HHMMSS")
#do_tests()
@write_fn("../tmp/test17_$today_str" * "_" * "$time_str.txt", do_tests())

