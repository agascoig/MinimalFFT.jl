
# Bluestein's FFT algorithm

mutable struct bs_buffer{T}
    a_n::Vector{T}
    b_n::Vector{T}
    A_X::Vector{T}
    B_X::Vector{T}
    M::Int64
    inverse::Bool
end

const bs_dict = IdDict{DataType, bs_buffer}()

function fft_bluestein!(y::Vector{T}, x::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    M = nextpow(2, 2 * N - 1)
    e1 = 63 - leading_zeros(M)

    @inbounds begin

        impiN = inverse ? 1.0im * pi / N : -1.0im * pi / N

        init::Bool = !haskey(bs_dict, T)
        if !init
            buff::bs_buffer{T} = bs_dict[T]
            if buff.M !== M
                init = true
            elseif buff.inverse !== inverse
                buff.b_n .= conj.(buff.b_n)
                buff.inverse = inverse
            end
        end

        if init
            b_n = zeros(T, M)
            b_n[1] = one(T)
            for n = 2:N
                c_e = exp(-impiN * (n-1) * (n-1))
                b_n[n] = c_e
                b_n[M-n+2] = c_e
            end
            buff = bs_buffer{T}(
                Vector{T}(undef, M), 
                b_n, 
                Vector{T}(undef, M),
                Vector{T}(undef, M), M, inverse)
            bs_dict[T] = buff
        end

        a_n::Vector{T} = buff.a_n
        a_n .= zero(T)

        b_n::Vector{T} = buff.b_n

        for n = 1:N
            c = conj(b_n[n])
            a_n[n] = x[bp+stride*(n-1)] * c
            y[bp+stride*(n-1)] = c
        end

        A_X::Vector{T} = buff.A_X
        B_X::Vector{T} = buff.B_X
        B_X .= b_n

        A_X, a_n = fftr2!(A_X, a_n, M, e1, 1, 1, false)
        B_X, a_n = fftr2!(a_n, B_X, M, e1, 1, 1, false)
        A_X .*= B_X
        B_X, A_X = fftr2!(B_X, A_X, M, e1, 1, 1, true)
        B_X .*= (1.0 / M)

        for i = 1:N
            y[bp+stride*(i-1)] *= B_X[i]
        end
    y, x
    end
end
