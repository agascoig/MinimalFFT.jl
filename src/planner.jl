
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
    ip = inner_plan()
    P.ipd[rd] = ip

    p_factors = factor(n)

    fn_mp = (ip, nf, b, e, f) -> begin
       push!(ip.ns, nf)
       push!(ip.bases, b)
       push!(ip.exponents, e)
       push!(ip.funs, f)
    end

    if n <= DIRECT_SZ
        fn_mp(ip, n, 1, n, direct_dft!)
    elseif (n & (n - 1)) == 0
        fn_mp(ip, n, 2, 63-leading_zeros(n), fftr2!) # power of 2
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
            fn_mp(ip, nf, b, e, f)
        end
    else
        fn_mp(ip, n, n, 1, fft_bluestein!)
    end
end

function gen_plan(P::MyPlan{T}) where {T}
    for r in P.region
        plan_1d(P, P.n[r], r)
    end
end

# the output size is always the same as the input size here
# y is output, x is unchanged
function execute_plan(P::MyPlan{U}, y::Vector{S}, x::Vector{T}, r::Int) where {U,S<:Complex,T<:Complex}
    inverse = bt(P, P_INVERSE)
    orig_y = y
    X = copy(x)

    ip = P.ipd[r]
    lf = length(ip.funs)
    if lf == 1
        y, X = ip.funs[1](y, x, ip.exponents[1], inverse)
    else
        es = ip.exponents
        ns = ip.ns
        fns = ip.funs
        inverse = bt(P, P_INVERSE)
        y, X = prime_factor!(y, X, es..., ns..., fns..., inverse)
    end

    if y !== orig_y
        orig_y .= y
    end
    orig_y, x
end

