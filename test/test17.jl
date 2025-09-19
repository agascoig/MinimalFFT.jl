
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

function stockham(Y, X, fn1, N, e1, bp, stride, inverse)
    Y, X = fn1(Y, X, N, e1, bp, stride, inverse)
    Y, X
end

function test_fft_kernel(repeat_count::Int64, Y_ref::Vector{T}, Y::Vector{T}, X_ref::Vector{T}, X::Vector{T}, copy_X::Vector{T},
    P_ref, P, bm, t_ref_s::Ref{Float64}, t_s::Ref{Float64}, Ns::Vector{Int64}, fns::Vector{U}, es::Vector{Int64},
    parent_fn::Function, bp::Int64, stride::Int64, inverse::Bool) where {T,U}

    X_ref .= copy_X
    X .= copy_X

    t = @timed P_ref * X_ref

    if t.time < 10e-6
        repeat_count *= 40 # oversample if less than 10 us
    end

    X_ref .= copy_X

    num_tests = repeat_count

    function run_ref(repeat_count, P_ref, Y_ref, X_ref, copy_X)
        for i = 1:repeat_count
            Y_ref .= P_ref * X_ref
            X_ref .= copy_X
        end
        nothing
    end

    t_ref = @timed run_ref(repeat_count, P_ref, Y_ref, X_ref, copy_X)

    function run_test(repeat_count, parent_fn, P, Y, X, copy_X, es, Ns, bp, stride, inverse)
        for i = 1:repeat_count
            if P !== nothing
                if P isa MinimalFFT.ScaledPlan
                    Y, X = MinimalFFT.execute_plan(P.p, Y, X, 1, bp, stride)
                else
                    Y, X = MinimalFFT.execute_plan(P, Y, X, 1, bp, stride)
                end
                X .= copy_X
            else
                if length(Ns) == 1
                    Y, X = parent_fn(Y, X, fns[1], prod(Ns), es[1], bp, stride, inverse)
                elseif length(Ns) == 2
                    Y, X = parent_fn(Y, X, es[1], es[2], Ns[1], Ns[2], fns[1], fns[2], bp, stride, inverse)
                elseif length(Ns) == 3
                    Y, X = parent_fn(Y, X, es[1], es[2], es[3], Ns[1], Ns[2], Ns[3], fns[1], fns[2], fns[3], bp, stride, inverse)
                else
                    Y, X = bluestein!(Y, X, prod(Ns), 1, 1, inverse)
                end
                X .= copy_X
            end
        end
        Y
    end

    t = @timed run_test(num_tests, parent_fn, P, Y, X, copy_X, es, Ns, bp, stride, inverse)

    Y .= t.value # in case flipped

    if bm != 0
        t_ref_s[] = t_ref.time / num_tests
        t_s[] = t.time / num_tests
    end
end

function test_fft(name::String, bm::Bool, inverse::Bool, N::Int64, pc::Ref{Int64},
    fc::Ref{Int64}, fn1::Function, P::Union{Nothing,MinimalFFT.MinimalPlan{T},MinimalFFT.ScaledPlan},
    Ns::Vector{Int64}, fns::Vector{U}, es::Vector{Int64}, bp::Int64, stride::Int64) where {T,U}

    repeat_count = 20

    @assert N===prod(Ns) "N must be the product of Ns"

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

    t_ref_s = Ref{Float64}(0.0)
    t_s = Ref{Float64}(0.0)

    test_fft_kernel(repeat_count, Y_ref, Y, X_ref, X, copy_X,
        P_ref, P, bm, t_ref_s, t_s, Ns, fns, es,
        fn1, bp, stride, inverse)

    if inverse
        Y .*= (1 / N)
    end

    if !(Y ≈ Y_ref)
        println("Failed for $name: N=$N=$Ns es=$es error=", sum(abs.(Y - Y_ref)), " time = ", @sprintf("%.2e", t_s[]), " fns=$fns")
        fc[] += 1
    else
        if bm
            println("Passed for $name: N=$N=$Ns es=$es time=", @sprintf("%.2e", t_s[]), " factor_ref=", @sprintf("%.3f", t_s[] / t_ref_s[]), " fns=$fns")
        else
            println("Passed for $name: N=$N=$Ns es=$es fns=$fns")
        end
        pc[] += 1
    end
end

function driver(d, radix::Vector{Int64}, bm::Bool, pc::Ref{Int64}, fc::Ref{Int64},
    inverse::Bool, parent_fn::Function, N_vals::Vector{Vector{Int64}}, name::String)

    bp = 1
    stride = 1

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
        v = eltype(d)((prod(NF), parent_fn, NF, es, inverse, bm))

        if i != length(NF) + 1 || (v in d)
            continue
        end

        push!(d, v)

        test_fft(name, bm, inverse, prod(NF), pc, fc, parent_fn, nothing,
            NF, fns, es, bp, stride)
    end
end

# test cases below

function do_tests()
    day = Dates.format(Dates.today(), "yyyy-mm-dd")
    t = Dates.format(Dates.now(), "HH:MM:SS")
    println("# $day time: $t\n\n")
    factor_1 = [[8], [25], [27], [16], [125], [49], [64], [81], [9 * 9 * 9], [256]]
    factor_2 = [[4, 25], [25, 4], [4, 49], [8, 9], [256, 25], [25, 256],
        [16, 5], [8, 7], [11, 8], [25, 4], [49, 3], [9, 8],
        [25, 256], [16, 5], [8, 7], [25, 4], [1, 256]]
    factor_3 = [[9, 5, 49], [9, 49, 5], [5, 9, 49], [49, 5, 9],
        [8, 7, 25], [7, 25, 8], [2, 3, 5], [2, 5, 3], [3, 2, 5],
        [3, 5, 2], [64, 3, 5], [3, 5, 64], [5, 64, 3], [1, 1, 64],
        [27, 625, 49], [625, 27, 49], [49, 27, 625], [49, 625, 27]]

    pc = Ref{Int64}(0)
    fc = Ref{Int64}(0)

    function level1_2_tests()
        # inverse, parent_fn, N_vals
        d = Set{Tuple{Int64,Function,Vector{Int64},Vector{Int64},Bool,Bool}}()

        #    driver(d, radix::Vector{Int64}, bm::Bool, pc::Ref{Int64}, fc::Ref{Int64}, 
        #inverse::Bool, parent_fn::Function, N_vals::Vector{Vector{Int64}}, name::String)

        bm = true
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 1")
        driver(d, [2, 9, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 2")
        driver(d, [4, 3, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 3")
        driver(d, [4, 9, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 4")
        driver(d, [8, 3, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 5")
        driver(d, [8, 9, 5, 7], bm, pc, fc, false, stockham, factor_1, "stockham test 6")

        empty!(d)

        driver(d, [2], bm, pc, fc, false, stockham, factor_1, "timed stockham test 0")
        driver(d, [3], bm, pc, fc, false, stockham, factor_1, "timed stockham test 1")
        driver(d, [4], bm, pc, fc, false, stockham, factor_1, "timed stockham test 2")
        driver(d, [5], bm, pc, fc, false, stockham, factor_1, "timed stockham test 3")
        driver(d, [7], bm, pc, fc, false, stockham, factor_1, "timed stockham test 4")
        driver(d, [8], bm, pc, fc, false, stockham, factor_1, "timed stockham test 5")
        driver(d, [9], bm, pc, fc, false, stockham, factor_1, "timed stockham test 6")

        empty!(d)

        driver(d, [2, 3, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 1")
        driver(d, [2, 9, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 2")
        driver(d, [4, 3, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 3")
        driver(d, [4, 9, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 4")
        driver(d, [8, 3, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 5")
        driver(d, [8, 9, 5, 7], bm, pc, fc, true, stockham, factor_1, "stockham inverse test 6")

        empty!(d)

        driver(d, [2, 3, 5, 7], bm, pc, fc, true, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 1 timed")
        driver(d, [4, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 2")
        driver(d, [8, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 3")
        driver(d, [2, 9, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 4")
        driver(d, [8, 9, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_2, "prime factor 2 test 5")

        empty!(d)

        driver(d, [2, 3, 5, 7], bm, pc, fc, true, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 1 timed")
        driver(d, [4, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 2")
        driver(d, [8, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 3")
        driver(d, [2, 9, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 4")
        driver(d, [8, 9, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_2, "mixed radix 2 test 5")

        empty!(d)

        driver(d, [2, 3, 5, 7], bm, pc, fc, true, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 1")
        driver(d, [4, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 2")
        driver(d, [8, 3, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 3")
        driver(d, [2, 9, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 4")
        driver(d, [8, 9, 5, 7], bm, pc, fc, false, MinimalFFT.prime_factor!, factor_3, "prime factor 3 test 5")

        empty!(d)

        driver(d, [2, 3, 5, 7], bm, pc, fc, true, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 0")
        driver(d, [2, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 1 timed")
        driver(d, [4, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 2")
        driver(d, [8, 3, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 3")
        driver(d, [2, 9, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 4")
        driver(d, [8, 9, 5, 7], bm, pc, fc, false, MinimalFFT.mixed_radix!, factor_3, "mixed radix 3 test 5")
    end

    level1_2_tests()

    function planner_tests()
        planner_n = [100, 196, 72, 6400, 80, 56, 100, 147, 72, 2205,
            1400, 30, 960, 826875, 2 * 3 * 5 * 7 * 11 * 13, 1 << 20, 1 << 22]

        planner_n_inverse = planner_n

        for n in planner_n
            P = MinimalFFT.MinimalPlan{ComplexF64}(ComplexF64, (n,), 1, MinimalFFT.P_NONE)
            Pinv = MinimalFFT.inv(P)

            IP = P.ipd[1] # first region
            NS = [ip.ns for ip in IP]
            FC = [ip.func for ip in IP]
            ES = [ip.exp for ip in IP]

            test_fft("execute_plan timed", true, false, n, pc, fc,
                MinimalFFT.execute_plan, P, NS, FC, ES, 1, 1) # planner_do_fft
            test_fft("execute_plan inverse timed", true, true, n, pc, fc,
                MinimalFFT.execute_plan, Pinv, NS, FC, ES, 1, 1) # planner_do_fft inverse
        end
    end

    planner_tests()

    test_fft("large power of 2 radix test 1", true, false, 2^20, pc, fc, stockham, nothing, [2^20], [MinimalFFT.fftr2!], [20], 1, 1)
    test_fft("large power of 2 radix test 2", true, true, 2^22, pc, fc, stockham, nothing, [2^22], [MinimalFFT.fftr2!], [22], 1, 1)

    println("$(pc[]) tests passed.")
    println("$(fc[]) tests failed.")
end

today_str = Dates.format(Dates.today(), "yyyymmdd")
time_str = Dates.format(Dates.now(), "HHMMSS")
do_tests()
#@write_fn("../profile/test17_$today_str" * "_" * "$time_str.txt", do_tests())

