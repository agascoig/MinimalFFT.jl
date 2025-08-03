
using Primes, FFTW, Random

import Base.@inbounds
macro inbounds(ex)
    esc(ex) # Disabled bounds checking
end

include("../src/indexer.jl")
include("../src/stockham.jl")
include("../src/bluestein.jl")
include("../src/mixedradix.jl")
include("../src/rader.jl")
include("../src/pfa.jl")
include("../src/direct.jl")
include("../src/planner.jl")

Random.seed!(6502)

fn_dict = Dict(2 => fftr2!,
    3 => fftr3!,
    4 => fftr4!,
    5 => fftr5!,
    7 => fftr7!,
    8 => fftr8!,
    9 => fftr9!,
    11 => fft_bluestein!,
    13 => fft_bluestein!
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

function check_fft(pc::Ref{Int64}, fc::Ref{Int64}, parent_fn::Function, fns, es, N_decomp, name)
    l = length(es)
    @assert l == length(fns) && l == length(N_decomp) "check_fft: length mismatch"

    N = prod(N_decomp)
    Y = zeros(ComplexF64, N)
    X = randn(ComplexF64, N)

    Y_fftw = fft(copy(X))

    if l == 1
        Y, X = fns[1](Y, copy(X), es..., false)
    else
        Y, X = parent_fn(Y, copy(X), es..., N_decomp..., fns..., false)
    end
    if !(Y ≈ Y_fftw)
        println("Failed for $name: N=$N=", N_decomp, " es=$es fns=$fns error=", sum(abs.(Y - Y_fftw)))
        fc[] += 1
    else
        println("Passed for $name: N=$N=", N_decomp)
        pc[] += 1
    end
end

function check_fft(radix::Vector{Int64}, pc::Ref{Int64}, fc::Ref{Int64}, parent_fn::Function,
    N_vals::Vector{Vector{Int64}}, name)

    for NF in N_vals
        bs = zeros(Int64, length(NF))
        es = zeros(Int64, length(NF))
        fns = Vector{Function}(undef, length(NF))
        i = 1
        for f in NF
            for j=1:length(radix)
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
        if i!=length(NF) + 1
            return
        end
        check_fft(pc, fc, parent_fn, fns, es, NF, name)
    end
end

function check_do_fft(radix::Vector{Int}, pc::Ref{Int64}, fc::Ref{Int64}, N)
    Y = zeros(ComplexF64, N)
    X = randn(ComplexF64, N)
    name = "do_fft"

    Y_fftw = fft(copy(X))

    Y, X = do_fft(Y, copy(X), 0, false)

    if !(Y ≈ Y_fftw)
        println("Failed for $name: N=$N error=", sum(abs.(Y - Y_fftw)))
        fc[] += 1
    else
        println("Passed for $name: N=$N")
        pc[] += 1
    end
end

# test cases below

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

ifn = identity

#=
check_fft([2,3,5,7], pc, fc, ifn, factor_1, "stockham test 1")
check_fft([2,9,5,7], pc, fc, ifn, factor_1, "stockham test 2")
check_fft([4,3,5,7], pc, fc, ifn, factor_1, "stockham test 3")
check_fft([4,9,5,7], pc, fc, ifn, factor_1, "stockham test 4")
check_fft([8,3,5,7], pc, fc, ifn, factor_1, "stockham test 5")
check_fft([8,9,5,7], pc, fc, ifn, factor_1, "stockham test 6")
=#

check_fft([2,3,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 1")
check_fft([4,3,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 2")
check_fft([8,3,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 3")
check_fft([2,9,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 4")
check_fft([4,9,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 5")
check_fft([8,9,5,7], pc, fc, prime_factor!, factor_2, "prime factor 2 test 6")

check_fft([2,3,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 1")
check_fft([4,3,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 2")
check_fft([8,3,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 3")
check_fft([2,9,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 4")
check_fft([4,9,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 5")
check_fft([8,9,5,7], pc, fc, mixed_radix!, factor_2, "mixed radix 2 test 6")

check_fft([2,3,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 1")
check_fft([4,3,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 2")
check_fft([8,3,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 3")
check_fft([2,9,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 4")
check_fft([4,9,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 5")
check_fft([8,9,5,7], pc, fc, prime_factor!, factor_3, "prime factor 3 test 6")

check_fft([2,3,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 1")
check_fft([4,3,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 2")
check_fft([8,3,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 3")
check_fft([2,9,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 4")
check_fft([4,9,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 5")
check_fft([8,9,5,7], pc, fc, mixed_radix!, factor_3, "mixed radix 3 test 6")

planner_n = [256, 3^5, 3^5*2^8, 7^2*2^5, 7^2*2^6*3^4, 2^5*5^4*3^2,
11^2 * 2^8, 13 * 8, 11 * 3^5, 11^2]

for n in planner_n
    check_do_fft(collect(keys(fn_dict)), pc, fc, n)
end

println("$(pc[]) tests passed.")
println("$(fc[]) tests failed.")
