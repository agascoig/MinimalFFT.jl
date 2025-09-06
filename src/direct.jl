
# For approx N<20, the direct DFT can be as fast as the FFT
# due to lower communication cost.  Also useful for testing.

mutable struct direct_buffer
    B::ComplexF64
    N::Int64
    inverse::Bool
end

const direct_buff::direct_buffer = direct_buffer(ComplexF64(0.0), 0, false)

function direct_dft!(y::Vector{T}, x::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
        B::T = zero(T)
        if direct_buff.N == N
            B = direct_buff.B
            if inverse != direct_buff.inverse
                B = conj(B)
                direct_buff.B = B
                direct_buff.inverse = inverse
            end
        else
            A = inverse ? 2im * pi / N : -2im * pi / N
            B = exp(A)
            direct_buff.B = B
            direct_buff.N = N
            direct_buff.inverse = inverse
        end

        W_step = one(T)
        for k = 1:N
            W = one(T)
            s = zero(T)
            for n = 1:N
                s = s + W * x[bp+stride*(n-1)]
                W = W * W_step
            end
            y[bp+stride*(k-1)] = s
            W_step = W_step * B
        end
        y, x
    end
end
