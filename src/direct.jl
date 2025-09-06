
# For approx N<20, the direct DFT can be as fast as the FFT
# due to lower communication cost.  Also useful for testing.

function direct_dft!(y::Vector{T}, x::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
        A = inverse ? 2im * pi / N : -2im * pi / N
        W_step = one(T)
        B = exp(A)
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
