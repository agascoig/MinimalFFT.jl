
using Primes

fftr_dict = Dict(   2 => fftr2!,
                    3 => fftr3!,
                    4 => fftr4!,
                    5 => fftr5!,
                    7 => fftr7!,
                    8 => fftr8!,
                    9 => fftr9!)

const DIRECT_SZ = 15

function plan_1d(P, n, rd)
    if haskey(P.ipd, rd)
        return
    end

    p_factors = factor(n)

    fn_mp = (P, nf, b, e, f) -> begin
        ip = inner_plan(nf, b, e, f)
        if !haskey(P.ipd, rd)
            P.ipd[rd] = Vector{inner_plan}()
        end
        push!(P.ipd[rd],ip)
    end

    if n <= DIRECT_SZ
        fn_mp(P, n, 1, n, direct_dft!)
    elseif (n & (n - 1)) == 0
        fn_mp(P, n, 2, 63-leading_zeros(n), fftr2!) # power of 2
    elseif length(p_factors) < 4
        l = length(p_factors)
        c = collect(p_factors)
        ns = Dict{Int64,Int64}()
        for i = 1:l
            nf = c[i][1]^c[i][2]
            ns[c[i][1]] = nf
        end
        sort!(c, by = x -> ns[x[1]], rev=true)
        for be in c
            b = be[1]
            e = be[2]
            nf = b^e
            f = (nf <= DIRECT_SZ) ? direct_dft! : 
            (haskey(fftr_dict, b) ? fftr_dict[b] : fft_bluestein!)
            fn_mp(P, nf, b, e, f)
        end
    else
        fn_mp(P, n, n, 1, fft_bluestein!)
    end
end

function gen_plan(P::MinimalPlan{T}) where {T}
    for r in P.region
        plan_1d(P, P.n[r], r)
    end
end

# the output size is always the same as the input size here
function execute_plan(P::MinimalPlan{U}, y::AbstractVector{S}, x::AbstractVector{T}, 
    r::Int64) where {U,S<:Complex,T<:Complex}
    inverse = bt(P, P_INVERSE)

    ipv = P.ipd[r]
    lf = length(ipv)
    if lf == 1
        y, x = ipv[1].fun(y, x, 1, 1, length(x), ipv[1].exp, inverse)
    elseif lf < 4
        es = [a.exp for a in ipv]
        ns = [a.ns for a in ipv]
        fns = [a.fun for a in ipv]
        y, x = prime_factor!(y, x, es..., ns..., fns..., inverse)
    else
        y, x = fft_bluestein!(y, x, 1, 1, length(x), 0, inverse)
    end

    (y, x)
end

