
# For approx N<20, the direct DFT can be as fast as the FFT
# due to lower communication cost.

function direct_dft!(y::Vector{T}, x::Vector{T}, e1::Int64, inverse::Bool) where {T<:Complex}
    N = length(x)
    A = inverse ? 2.0im * pi / N : -2.0im * pi / N
    y .= 0.0 + 0.0im
    W_step = 1.0 + 0.0im
    B = exp(A)
    for k = 1:N
        W = 1.0 + 0.0im
        for n = 1:N
            @inbounds y[k] += x[n] * W
            W *= W_step
        end
        W_step *= B
    end
    y, x
end
