# Licnese is MIT
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The purpose of this MinimalFFT module is to provide a basic
# implementation of the AbstractFFTs FFT interface.
#
# References
# [1] Takahasi, Daisuke.  "Fast Fourier Transform Algorithms for Parallel Computers"
#     Springer Nature, Oct 5, 2019.
# [2] "Chirp Z-transform, Bluestein's algorithm." Wikipedia.  Accessed May 9th, 2024.
# [3] Saad Bouguezel, M. Omair Ahmad, M.N.S. Swamy. "An Alternate Approach for
#     Developing Higher Radix FFT Algorithms."  APCCAS 2006.
# [4] Van Loan, Charles.  "Computational Frameworks for the Fast Fourier Transform."
#     SIAM.  1992.

module MinimalFFT

using Reexport

@reexport using AbstractFFTs

import Base: *, size
import AbstractFFTs: Plan, ScaledPlan, plan_fft, plan_fft!, plan_bfft, plan_bfft!,
    plan_ifft, plan_ifft!, fftdims, plan_inv, inv,
    plan_rfft, plan_irfft, plan_brfft,
    AdjointStyle, AdjointPlan, FFTAdjointStyle, RFFTAdjointStyle, IRFFTAdjointStyle
import LinearAlgebra: mul!, rmul!, lmul!

# TBD: print plan
## Define how MyType is printed in standard output
#function Base.show(io::IO, x::MyType)
#    print(io, "MyType(name: ", x.name, ", value: ", x.value, ")")
#end

include("plan.jl")

function min_plan(S::Type, D::Type, x, region, flags)
    sx = size(x)
    if !bt(flags, P_REAL) || !bt(flags, P_INVERSE)
        flags |= sx[first(region)] & 1 == 1 ? P_ODD : P_NONE # set if needed, if not plan_irfft or plan_brfft
    end
    P = MinimalPlan{S}(D, size(x), region, flags)
    @assert !(bt(P, P_REAL) && bt(P, P_INPLACE)) "real inplace plan not supported"
    if bt(P, P_SCALED)
        ScaledPlan(P, scaling_factor(P))
    else
        P
    end
end

plan_fft(x, region; kws...) = min_plan(eltype(x), eltype(x), x, region, P_NONE)
plan_fft!(x, region; kws...) = min_plan(eltype(x), eltype(x), x, region, P_INPLACE)
plan_ifft(x::Array{T,N}, region; kws...) where {T<:Number,N} = min_plan(eltype(x), eltype(x), x, region, P_INVERSE | P_SCALED)
plan_ifft!(x::Array{T,N}, region; kws...) where {T<:Number,N} = min_plan(eltype(x), eltype(x), x, region, P_INVERSE | P_INPLACE | P_SCALED)

# bfft: ifft but unscaled
plan_bfft(x, region; kws...) = min_plan(eltype(x), eltype(x), x, region, P_INVERSE | P_ISBFFT)
plan_bfft!(x, region; kws...) = min_plan(eltype(x), eltype(x), x, region, P_INVERSE | P_INPLACE | P_ISBFFT)

# rfft, irfft, brfft
plan_rfft(x::Array{T,N}, region; kws...) where {T<:Real,N} = min_plan(T, Complex{T}, x, region, P_REAL)
plan_rfft(x::Array{T,N}, region; kws...) where {T<:Complex,N} = min_plan(real(T), T, x, region, P_REAL) # force real source type
plan_irfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Complex,N} = min_plan(T, real(T), x, region, P_INVERSE | P_REAL | P_SCALED | (d & 1 == 1 ? P_ODD : P_NONE))
plan_brfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Complex,N} = min_plan(T, real(T), x, region, P_INVERSE | P_REAL | P_ISBFFT | (d & 1 == 1 ? P_ODD : P_NONE))

# Adjoint support
function AdjointStyle(P::MinimalPlan{T}) where {T}
    if bt(P, P_REAL)
        if bt(P, P_INVERSE)
            return IRFFTAdjointStyle(out_N_irfft(P)) #d: output size
        else
            return RFFTAdjointStyle()
        end
    end
    FFTAdjointStyle()
end

size(P::MinimalPlan{T}) where {T<:Number} = P.n # the FFT input size

out_N_rfft(P::MinimalPlan{T}) where {T<:Number} = (P.n[first(P.region)] ÷ 2) + 1
out_N_irfft(P::MinimalPlan{T}) where {T<:Number} = (P.n[first(P.region)] << 1) - 2 + (bt(P, P_ODD))

function mul!(y::Array{R,D}, P::MinimalPlan{S}, x::Array{T,E}) where {R<:Number,S<:Number,T<:Number,D,E}
    @assert P.n === size(x) "The plan input size must match input x dimensions."
    if bt(P, P_REAL) && bt(P, P_INVERSE) # irfft
        dim = first(P.region)
        C = selectdim(x, dim, size(x)[dim]-(!bt(P, P_ODD)):-1:2)
        x = cat(x, conj(C), dims=dim)
    end

    irfft = bt(P, P_REAL) && bt(P, P_INVERSE)
    rfft = !bt(P, P_INVERSE) && bt(P, P_REAL)

    # x was real only if not inplace: make complex
    ix = eltype(x) <: Complex ? copy(x) : complex(x)
    oy = irfft || rfft ? zeros(eltype(ix), size(ix)) : y

    if D == 1
        oy, ix = execute_plan(P, oy, ix, 1)
    else
        soy = size(oy)
        ET = eltype(oy)
        for r in P.region
            oy, ix = do_fft_planned(P, oy, ix, r)
            ix = oy
        end
    end

    if irfft
        y .= real(oy) # force a real output
    elseif rfft
        rdim = first(P.region)
        y .= selectdim(oy, rdim, 1:out_N_rfft(P)) # truncate rfft output
    elseif oy !== y
        y .= oy
    end

    if bt(P, P_INPLACE)
        x .= bt(P, P_ISBFFT) ⊻ bt(P, P_INVERSE) ? scaling_factor(P) * y : y
    end

    y
end

function scaling_factor(P::MinimalPlan{T}) where {T<:Number}
    if bt(P, P_REAL) && bt(P, P_INVERSE)
        # get the full length of the output
        sz = AbstractFFTs.brfft_output_size(P.n, out_N_irfft(P), P.region)
    else
        sz = P.n # same as the input size for the purposes of scaling
    end
    rscale = real(P.D)
    s = rscale(1.0)
    for i in P.region
        s *= sz[i]
    end
    inv(s)
end

function plan_inv(P::MinimalPlan{T}) where {T<:Number}
    # note that ScaledPlan is immutable
    S = P.D

    if bt(P, P_REAL)
        nn = [P.n...]
        # get the new input size from the old output size
        os = bt(P, P_INVERSE) ? out_N_irfft(P) : out_N_rfft(P)
        nn[first(P.region)] = os
        nnt = Tuple(nn)
    else
        nnt = P.n
    end

    # no x available, so don't call min_plan and construct here
    IP = MinimalPlan{S}(T, nnt, P.region, P.flags ⊻ P_INVERSE)
    ScaledPlan{S}(IP, scaling_factor(IP))
end

# utility functions for output
function get_output_size(P::MinimalPlan{T}) where {T<:Number}
    if bt(P, P_REAL)
        s = bt(P, P_INVERSE) ? AbstractFFTs.brfft_output_size(P.n, out_N_irfft(P), P.region) :
            AbstractFFTs.rfft_output_size(P.n, P.region)
        return s
    end
    P.n
end

function output_buffer(P::MinimalPlan{T}) where {T<:Number}
    s = get_output_size(P)
    zeros(P.D, s)
end

# * operator
function *(P::MinimalPlan{T}, x::Array{T,N}) where {T<:Number,N}
    y = output_buffer(P)
    mul!(y, P, x)
    y
end

include("indexer.jl")
include("stockham.jl")
include("direct.jl")
include("mixedradix.jl")
include("pfa.jl")
include("bluestein.jl")
include("rader.jl")
include("planner.jl")

end # module
