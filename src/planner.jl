
using Primes

fftr_dict = Dict(2 => fftr2!,
    3 => fftr3!,
    4 => fftr4!,
    5 => fftr5!,
    7 => fftr7!,
    8 => fftr8!,
    9 => fftr9!)

const DIRECT_SZ = 16

# the output size is always the same as the input size here
# y is output, x is unchanged
function do_fft(y::Vector{S}, x::Vector{T}, unused::Int64, inverse::Bool) where {S<:Complex,T<:Complex}
    orig_y = y
    X = copy(x)
    N = length(x)
    if N < DIRECT_SZ
        y, X = direct_dft!(y, X, 0, inverse)
    elseif (N & (N - 1)) == 0
        y, X = fftr2!(y, X, 63 - leading_zeros(N), inverse)
    else
        factors = factor(N)
        l = length(factors)
        fn = () -> fft_bluestein!(y,X,0,inverse)
        if l == 1
            f = factors.pe[1]
            b1 = f[1]
            e1 = f[2]
            if haskey(fftr_dict, b1)
                fn = () -> (b1^e1 < DIRECT_SZ) ? direct_dft!(y,X,0,inverse) :
                fftr_dict[b1](y,X,f[2],inverse)
            end
            y, X = fn()
        elseif l == 2 || l == 3
            fs = [factors.pe[i] for i in 1:l]
            bs = [a[1] for a in fs]
            es = [a[2] for a in fs]
            NS = [bs[i]^es[i] for i in 1:l]
            fns = [haskey(fftr_dict, bs[i]) ? fftr_dict[bs[i]] : 
               (NS[i] < DIRECT_SZ ? direct_dft!(y,X,0,inverse) : fft_bluestein!) for i in 1:l]
            fn = () -> prime_factor!(y, X, es..., NS..., fns..., inverse)
            y, X = fn()
        else
            y, X = fn()
        end
    end
    if y !== orig_y
        orig_y .= y
    end
    orig_y, x
end
