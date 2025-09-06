
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
