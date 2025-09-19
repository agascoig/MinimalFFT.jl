
# plan.jl

using Primes

const P_NONE = 0
const P_INVERSE = 1
const P_INPLACE = 2
const P_REAL = 4
const P_ISBFFT = 8
const P_ODD = 16
const P_SCALED = 32

# inner_plan for a region
struct inner_plan
    ns::Int64
    base::Int64
    exp::Int64
    func::Function
end

r2fun = Dict{Int64,Function}(2=>fftr2!, 3=>fftr3!, 4=>fftr4!,
    5=>fftr5!, 7=>fftr7!, 8=>fftr8!, 9=>fftr9!)

mutable struct MinimalPlan{T} <: Plan{T}
    D::DataType # destination type, for real fft     # required by AbstractFFTs
    n::Tuple{Vararg{Int64}} # Size of the FFT input     # required by AbstractFFTs
    region::Union{Int,UnitRange{Int64}}     # required by AbstractFFTs
    flags::Int32 # bit vector of fft type
    os::Tuple{Vararg{Int64}} # output size
    ipd::Dict{Int64,Vector{inner_plan}} # region -> inner_plan

    pinv::ScaledPlan # required by AbstractFFTs

    MinimalPlan{T}(D, n, region, flags) where {T} =
        begin
            if !(D <: AbstractFloat) || (D <: Complex && !(real(D) <: AbstractFloat))
                D=float(D)
            end
            mp = new(D, n, region, flags, Tuple{Vararg{Int64}}(()), Dict{Int64,Vector{inner_plan}}())
            mp.os = get_output_size(mp)
            gen_inner_plan(mp)
            mp
        end
end

bt(flags, flag) = flags & flag != 0 ? true : false
bt(P::MinimalPlan{T}, flag) where {T<:Number} = bt(P.flags, flag)

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
        push!(P.ipd[rd], ip)
    end

    if n <= DIRECT_SZ
        fn_mp(P, n, n, 1, direct_dft!)
    elseif (n & (n - 1)) == 0
        fn_mp(P, n, 2, 63 - leading_zeros(n), fftr2!) # power of 2
    elseif length(p_factors) < 4
        l = length(p_factors)
        c = collect(p_factors)
        ns = Dict{Int64,Int64}()
        for i = 1:l
            nf = c[i][1]^c[i][2]
            ns[c[i][1]] = nf
        end
        sort!(c, by=x -> ns[x[1]], rev=true)
        for be in c
            b = be[1]
            e = be[2]
            nf = b^e
            f = (nf <= DIRECT_SZ) ? direct_dft! :
                (haskey(r2fun, b) ? r2fun[b] : fft_bluestein!)
            fn_mp(P, nf, b, e, f)
        end
    else
        fn_mp(P, n, n, 1, fft_bluestein!)
    end
end

function gen_inner_plan(P::MinimalPlan{T}) where {T}
    for r in P.region
        if r == first(P.region) && bt(P, P_REAL) && bt(P, P_INVERSE)
            nt = out_N_irfft(P)
        else
            nt = P.n[r]
        end
        plan_1d(P, nt, r)
    end
end

const dispatch = [fftr2!, fftr3!, fftr4!, fftr5!, fftr7!, 
                  fftr8!, fftr9!, direct_dft!, fft_bluestein!]

# the output size is always the same as the input size here
function execute_plan(P::MinimalPlan{U}, y::Vector{S}, x::Vector{T},
    r::Int64, bp::Int64, instride::Int64) where {U,S<:Complex,T<:Complex}
    @inbounds begin
        inverse = bt(P, P_INVERSE)
        ipv = P.ipd[r]
        lf = length(ipv)
        ipv1 = ipv[1]
        fn1 = ipv1.func
        if lf == 1 || lf>3
            y, x = fn1(y, x, ipv1.ns, ipv1.exp, bp, instride, inverse)
        elseif lf == 2
            ipv2=ipv[2]
            fn2 = ipv2.func
            y, x = prime_factor!(y, x, ipv1.exp, ipv2.exp, ipv1.ns, ipv2.ns,
                fn1, fn2, bp, instride, inverse)
        elseif lf == 3
            ipv2=ipv[2]
            ipv3=ipv[3]
            fn2 = ipv2.func
            fn3 = ipv3.func
            y, x = prime_factor!(y, x, ipv1.exp, ipv2.exp, ipv3.exp,
                ipv1.ns, ipv2.ns, ipv3.ns, fn1, fn2, fn3,
                bp, instride, inverse)
        end
        (y, x)
    end
end

