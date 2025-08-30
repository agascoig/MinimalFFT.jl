
# For approx N<20, the direct DFT can be as fast as the FFT
# due to lower communication cost.  Also useful for testing.

function direct_dft!(y::AbstractVector{T}, x::AbstractVector{T}, 
    bp::Int64, stride::Int64, N::Int64, e1::Int64, inverse::Bool) where {T<:Complex}
    A = inverse ? 2im * pi / N : -2im * pi / N
    W_step = 1.0 + 0.0im
    B = exp(A)
    for k = 1:N
        W = 1.0 + 0.0im
        @inbounds y[bp+stride*(k-1)] = 0.0 + 0.0im
        for n = 1:N
            @inbounds y[bp+stride*(k-1)] += x[bp+stride*(n-1)] * W
            W *= W_step
        end
        W_step *= B
    end
    y, x
end
