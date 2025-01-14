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
# end of terms and conditions
#
# Please see THIRDPARTY.md for license information for other software used in this project.
#
# The purpose of this MinimalFFT module is to provide a basic
# implementation of the AbstractFFTs FFT interface.
#

module MinimalFFT

using Reexport

@reexport using AbstractFFTs

import Base: *, size
import AbstractFFTs: Plan, ScaledPlan, plan_fft, fft, plan_fft!, fft!, plan_bfft, plan_bfft!,
                     plan_ifft, plan_ifft!, ifft, ifft!, fftdims, plan_inv, inv,
                     plan_rfft, plan_irfft, plan_brfft, rfft, irfft,
                     AdjointStyle, AdjointPlan, FFTAdjointStyle, RFFTAdjointStyle, IRFFTAdjointStyle
import LinearAlgebra: mul!, rmul!

using Infiltrator # TBD: remove

mutable struct MyPlan{T} <: Plan{T}
    n::Tuple{Vararg{Int}} # Size of the FFT input
    inverse::Bool
    inplace::Bool
    real::Bool
    isbfft::Bool
    d::Int # for inverse real fft
    region::Union{Int, UnitRange{Int}}
    D::Type # destination type, for real fft 
    pinv::ScaledPlan # keep #undef, apparently does not work in AbstractFFTs
    MyPlan{T}(n, inverse, inplace, real, isbfft, d, region, D) where {T} = new(n,inverse,inplace,real,isbfft,d,region,D)
    function MyPlan{T}(o::MyPlan{T}) where {T}
        if isdefined(o,:piv)
            new(o.n,o.inverse,o.inplace,o.real,o.isbfft,
            o.d,o.region,o.D,o.pinv)
        else
            new(o.n,o.inverse,o.inplace,o.real,o.isbfft,
            o.d,o.region,o.D) # leave pinv undefined
        end
    end
end

function my_plan(S::Type,D::Type,x,inverse,inplace,real,isbfft,d,region,scaled_plan)
    if d==0
        d=size(x)[first(region)]
    end
    p=MyPlan{S}(size(x),inverse,inplace,real,isbfft,d,region,D)
    if scaled_plan
        ScaledPlan(p,scaling_factor(p))
    else
        p
    end
end

plan_fft(x, region; kws...) = my_plan(eltype(x),eltype(x),x,false,false,false,false,0,region,false)
plan_fft!(x, region; kws...) = my_plan(eltype(x),eltype(x),x,false,true,false,false,0,region,false)
plan_ifft(x::Array{T,N}, region; kws...) where {T<:Number,N} = my_plan(eltype(x),eltype(x),x,true,false,false,false,0,region,true)
plan_ifft!(x::Array{T,N}, region; kws...) where {T<:Number,N} = my_plan(eltype(x),eltype(x),x,true,true,false,false,0,region,true)

# bfft: ifft but unscaled
plan_bfft(x, region; kws...) = my_plan(eltype(x),eltype(x),x,true,false,false,true,0,region,false)
plan_bfft!(x, region; kws...) = my_plan(eltype(x),eltype(x),x,true,true,false,true,0,region,false)

# rfft, irfft, brfft
plan_rfft(x::Array{T,N}, region; kws...) where {T<:Real,N} = my_plan(T,Complex{T},x,false,false,true,false,0,region,false)
plan_rfft(x::Array{T,N}, region; kws...) where {T<:Complex,N} = my_plan(real(T),T,x,false,false,true,false,0,region,false) # force real source type
#plan_irfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Real,N} = my_plan{Complex{T},T}(x,false,false,true,false,d,region,true) # need !isbfft
plan_irfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Complex,N} = my_plan(T,real(T),x,true,false,true,false,d,region,true)
#plan_brfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Real,N} = my_plan{Complex{T},T}(x,true,false,true,true,d,region,false)
plan_brfft(x::Array{T,N}, d::Integer, region; kws...) where {T<:Complex,N} = my_plan(T,real(T),x,true,false,true,true,d,region,false)

# Adjoint support
function AdjointStyle(p::MyPlan{T}) where {T}
    if p.real
        if p.inverse
            return IRFFTAdjointStyle(p.d)
        else
            return RFFTAdjointStyle()
        end
    end
    FFTAdjointStyle()
end

function size(p::MyPlan{T}) where {T<:Number}
    p.n # the FFT input size
end

# mul! routines.  inner routines handle complex conversion.

function get_fft_fn(x::Vector{R},p::MyPlan{S}) where {R<:Number,S<:Number}
    f_table = [()->inner_fft(x),
               ()->inner_ifft(x,false),
               ()->inner_fft!(x),
               ()->inner_ifft!(x,false),
               ()->inner_rfft(x),
               ()->real(inner_irfft(x,p.d)),
               ()->error("no real inplace fft support"),
               ()->error("no real inplace fft support")]
    a = p.real ? 4 : 0
    b = p.inplace ? 2 : 0
    c = p.inverse ? 1 : 0
    f_table[a+b+c+1]
end

function mul!(y::Vector{R}, p::MyPlan{S}, x::Vector{T}) where {R<:Number,S<:Number, T<:Number}
    # output y is pre-allocated
    fn = get_fft_fn(x,p)
    y .= fn()

    if p.inplace
        s = p.isbfft ⊻ p.inverse ? scaling_factor(p) : 1.0
        x .= s*y
    end
    y
end

function mul!(y::Matrix{R}, p::MyPlan{S}, x::Matrix{T}) where {R<:Number,S<:Number,T<:Number}
    region = p.region

    do_row = 2 in region
    do_column = 1 in region

    P = MyPlan{S}(p) # make a copy, and force not inplace
    P.inplace = false

    rows, columns = size(p)
    d = ndims(x)

    input = x

    if do_row
        yr = zeros(T, columns)
        fn = get_fft_fn(yr,P)
        for i = 1:rows
            yr .= input[i, :]
            a = fn()
            y[i, :] .= a
        end
        input = y
    end

    if do_column
        yc = zeros(T, rows)
        fn = get_fft_fn(yc,P)
        for i = 1:columns
            yc .= input[:, i]
            a = fn()
            y[:, i] .= a
        end
    end

    if p.inplace
        s = p.isbfft ⊻ p.inverse ? scaling_factor(p) : 1.0
        x .= s*y
    end
    y
end

function mul!(y::Array{R,3}, p::MyPlan{S}, x::Array{T,3}) where {R<:Number, S<:Number, T<:Number}
    P = MyPlan{S}(p) # make a copy, and force not inplace
    P.inplace = false
    region = p.region

    imax, jmax, kmax = size(p)

    input = x

    U = (T<:Complex) ? T : Complex{T}

    if 3 in region
        y1d = zeros(U, kmax)
        fn=get_fft_fn(y1d,P)
        for i = 1:jmax
            for j = 1:jmax
                y1d .= input[i, j, :]
                a = fn()
                y[i, j, :] .= a
            end
        end
        input = y
    end

    if 2 in region
        y1d = zeros(U, jmax)
        fn=get_fft_fn(y1d,P)
        for i = 1:jmax
            for k = 1:kmax
                y1d .= input[i, :, k]
                a = fn()
                y[i, :, k] .= a
            end
        end
        input = y
    end

    if 1 in region
        y1d = zeros(U, imax)
        fn=get_fft_fn(y1d,P)
        for j = 1:jmax
            for k = 1:kmax
                y1d .= input[:, j, k]
                a = fn()
                y[:, j, k] .= a
            end
        end
    end

    if p.inplace
        s = p.isbfft ⊻ p.inverse ? scaling_factor(p) : 1.0
        x .= s*y
    end
    y
end

function scaling_factor(p::MyPlan{T}) where {T<:Number}
    if p.real && p.inverse
        # get the full length of the output
        sz=AbstractFFTs.brfft_output_size(p.n,p.d,p.region)
    else
        sz=p.n # same as the input size for the purposes of scaling
    end
    s=1.0
    for i in p.region
        s*=sz[i]
    end
    inv(s)
end

function plan_inv(p::MyPlan{T}) where {T<:Number}
    # note that ScaledPlan is immutable
    S = p.D
    
    if p.real
        nn=[p.n...]
        # if real inverse, will have half the input size
        nn[first(p.region)] = p.inverse ? p.d : (nn[first(p.region)] ÷ 2) + 1
        nnt=Tuple(nn)
        ip=MyPlan{S}(nnt,!p.inverse,p.inplace,p.real,p.isbfft,p.d,p.region,T)
        return ScaledPlan{S}(ip,scaling_factor(ip))    
    end

    ip=MyPlan{S}(p.n,!p.inverse,p.inplace,p.real,p.isbfft,p.d,p.region,T)
    ScaledPlan{S}(ip,scaling_factor(ip))
end

# utility functions for output

function get_output_size(p::MyPlan{T}) where {T<:Number}
    if p.real
        s = p.inverse ? AbstractFFTs.brfft_output_size(p.n,p.d,p.region) :
        AbstractFFTs.rfft_output_size(p.n,p.region)
        return s
    end
    p.n
end

function output_buffer(::Type{S},p::MyPlan{T}) where {S<:Number,T<:Number}
    s = get_output_size(p)
    zeros(S, s)
end

# * operators

function *(p::MyPlan{T}, x::Array{T,N}) where {T<:Real,N}
    y = output_buffer(Complex{T},p)
    mul!(y, p, convert(Array{Complex{T},N},x))
end

function *(p::MyPlan{T}, x::Array{T,N}) where {T<:Complex,N}
    y = output_buffer(T, p)
    mul!(y, p, x)
    if p.real && p.inverse && !(p.D<:Complex)
        return real(y)
    end
    y 
end

# inner fft, ifft dispatch routines

function inner_fft(x::Vector{Complex{T}}) where {T<:Real}
    N = length(x)
    X = copy(x)
    inner_fft!(X)
    X
end

function inner_fft!(x::Vector{Complex{T}}) where {T<:Real}
    N = length(x)
    if N<16
        direct_dft!(x,false)
    elseif (N & (N-1))==0
        fftr2!(x,false)
    else
        fft_bluestein!(x,false)
    end
    x
end

function inner_ifft!(x::Vector{Complex{T}}, scale=true) where {T<:Real}
    N = length(x)
    if N<16
        direct_dft!(x,true)
    elseif (N & (N-1))==0
        fftr2!(x,true)
    else
        fft_bluestein!(x,true)
    end
    if scale
        x.= x/N
    end
    x
end

function inner_ifft(X::Vector{Complex{T}}, scale=true) where {T<:Real}
    N = length(X)
    x = copy(X)
    inner_ifft!(x,scale)
    x
end

function inner_rfft(x::Vector{T}) where {T<:Number}
    X = convert(Vector{ComplexF64}, x) # TBD: more specific cases
    N = length(x)
    if N<16
        direct_dft_real!(X,false)
    elseif (N & (N-1))==0
        fftr2_real!(X,false)
    else
        fft_bluestein_real!(X,false)
    end
    X[1:(size(x)[1] ÷ 2) + 1]
end

function inner_irfft(X::Vector{Complex{T}},d,scale=false) where {T<:Number}
    x = vcat(X,conj(X[end-((d+1)&1):-1:2]))
    N = length(x)
    if N<16
        direct_dft!(x, true)
    elseif (N & (N-1))==0
        fftr2!(x,true)
    else
        fft_bluestein!(x,true)
    end
    if scale
        x .= x/N
    end
    x
end

# overrides for AbstractFFTs.(fft,fft!,ifft,ifft!,rfft,irfft)

fft(x::Vector{Complex{T}}) where {T<:Real} = inner_fft(x)

fft!(x::Vector{Complex{T}}) where {T<:Real} = inner_fft!(x)

ifft(X::Vector{Complex{T}}) where {T<:Real} = inner_ifft(X, true)

ifft!(x::Vector{Complex{T}}) where {T<:Real} = inner_ifft!(x, true) 

rfft(x::Vector{T}, region) where {T<:Number} = inner_rfft(x)[1:(size(x)[1] ÷ 2) + 1]

irfft(X::Vector{Complex{T}}, d::Integer) where {T<:Real} = inner_irfft(X, d)

# MinimalFFT FFT and IFFT routines

function bitrev!(x)
    N = length(x)
    shamt = leading_zeros(N) + 1
    for n = 0:N-1
        r = bitreverse(n)
        r = r >>> shamt
        if n < r
            @inbounds x[n+1], x[r+1] = x[r+1], x[n+1]
        end
    end
    x
end

function fftr2(x::Vector{T}, invert=false) where {T<:Complex}
    X = copy(x)
    fftr2!(X,invert)
    X
end

function fftr2!(X::Vector{T}, invert=false) where {T<:Complex}
    N = length(X)
    
    @assert (N!=0 && (N & (N-1))==0) "fftr2!: N=$N, vector length must be power of 2"
    
    bitrev!(X)
    
    n = 1
    
    e_pi = invert ? 1.0im * pi : -1.0im * pi
    
    while n<N
        w = one(T)
        w_n = exp(e_pi / n)
        for k = 1:n
            i0 = k
            i1 = i0 + n
            while i1<=N
                @inbounds y0 = X[i0]
                @inbounds y1 = X[i1] * w
                @inbounds X[i0] = y0 + y1
                @inbounds X[i1] = y0 - y1
                i0 += (n << 1)
                i1 += (n << 1)
            end
            w = w * w_n
        end
        n = n << 1
    end
    X
end

function fftr2_real!(X::Vector{T}, invert=false) where {T<:Complex}
#TBD: possibly modify for real case
    N = length(X)
    
    @assert (N!=0 && (N & (N-1))==0) "fftr2_real!: N=$N, vector length must be power of 2"
    
    bitrev!(X)
    
    n = 1
    
    e_pi = invert ? 1.0im * pi : -1.0im * pi
    
    while n<N
        w = one(T)
        w_n = exp(e_pi / n)
        for k = 1:n
            i0 = k
            i1 = i0 + n
            while i1<=N
                @inbounds y0 = X[i0]
                @inbounds y1 = X[i1] * w
                @inbounds X[i0] = y0 + y1
                @inbounds X[i1] = y0 - y1
                i0 += (n << 1)
                i1 += (n << 1)
            end
            w = w * w_n
        end
        n = n << 1
    end
    X
end

function fft_bluestein!(x::Vector{T}, invert=false) where {T<:Complex}
    N = length(x)
    M = nextpow(2, 2*N-1)
    
    a_n = zeros(T, M)
    b_n = zeros(T, M)
    
    impiN = invert ? 1.0im*pi/N : -1.0im*pi/N
    
    a_n[1]=x[1]
    b_n[1]=1
    for n=1:N-1
        e=exp(impiN*n*n)
        a_n[n+1]=x[n+1]*e
        b_n[n+1]=conj(e)
        b_n[M-n+1]=b_n[n+1]
    end
    
    # convolve a_n with b_n
    A_X = fftr2!(a_n,false)
    B_X = fftr2(b_n,false)
    A_X .*= B_X
    fftr2!(A_X, true)
    scale = 1.0/M
    A_X .= A_X*scale
    x .= conj(b_n[1:N]).*A_X[1:N]
    x
end

function fft_bluestein_real!(x::Vector{T}, invert=false) where {T<:Complex}
    # TBD: specialize for the real case
    N = length(x)
    M = nextpow(2, 2*N-1)
    
    a_n = zeros(T, M)
    b_n = zeros(T, M)
    
    impiN = invert ? 1.0im*pi/N : -1.0im*pi/N
    
    a_n[1]=x[1]
    b_n[1]=1
    for n=1:N-1
        e=exp(impiN*n*n)
        a_n[n+1]=x[n+1]*e
        b_n[n+1]=conj(e)
        b_n[M-n+1]=b_n[n+1]
    end
    
    # convolve a_n with b_n
    A_X = fftr2!(a_n,false)
    B_X = fftr2(b_n,false)
    A_X .*= B_X
    fftr2!(A_X, true)
    scale = 1.0/M
    A_X .= A_X*scale
    x .= conj(b_n[1:N]).*A_X[1:N]
    x 
end

# For approx N<20, the direct DFT can be as fast as the FFT
# due to lower communication cost.
function direct_dft!(x::Vector{T}, invert=false) where {T<:Complex}
    N = length(x)
    X = zeros(T, size(x))
    A = invert ? 2.0im*pi/N : -2.0im*pi/N
    for k=1:N
        W_step = exp(A*(k-1)) # faster than sincos
        W = 1.0+0.0im
        for n=1:N
            @inbounds X[k]+=x[n]*W
            W *= W_step
        end
    end
    x .= X
end

function direct_dft_real!(x::Vector{T}, invert=false) where {T<:Complex}
    N = length(x)
    X = zeros(T, size(x))
    A = invert ? 2.0im*pi/N : -2.0im*pi/N
    L = (N ÷ 2) + 1
    for k=1:L
        W_step = exp(A*(k-1)) # faster than sincos
        W = 1.0+0.0im
        for n=1:N
            @inbounds X[k]+=x[n]*W
            W *= W_step
        end
    end
    x .= X
end

end # module
