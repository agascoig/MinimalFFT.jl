
# plan.jl

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
    fun::Function
end

mutable struct MyPlan{T} <: Plan{T}
    D::Type # destination type, for real fft     # required by AbstractFFTs
    n::Tuple{Vararg{Int}} # Size of the FFT input     # required by AbstractFFTs
    region::Union{Int,UnitRange{Int}}     # required by AbstractFFTs
    flags::Int32 # bit vector of fft type
    ipd::Dict{Int64,Vector{inner_plan}} # region -> inner_plan

    pinv::ScaledPlan # required by AbstractFFTs

    MyPlan{T}(D, n, region, flags) where {T} =
        begin
            mp = new(D, n, region, flags, Dict{Int64,Vector{inner_plan}}())
            gen_plan(mp)
            mp
        end
end

bt(flags, flag) = flags & flag != 0 ? true : false
bt(P::MyPlan{T}, flag) where {T<:Number} = bt(P.flags, flag)
