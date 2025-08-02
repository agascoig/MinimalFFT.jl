
# Bluestein's FFT algorithm

function fft_bluestein!(y::Vector{T}, x::Vector{T}, e1::Int64, inverse::Bool) where {T<:Complex}
    N = length(x)
    M = nextpow(2, 2 * N - 1)
    e1 = 63 - leading_zeros(M)

    a_n = zeros(T, M)
    b_n = zeros(T, M)

    impiN = inverse ? 1.0im * pi / N : -1.0im * pi / N

    @inbounds a_n[1] = x[1]
    @inbounds b_n[1] = 1.0
    @inbounds y[1] = 1.0
    for n = 2:N
        e = exp(impiN * (n-1) * (n-1))
        c_e = conj(e)
        @inbounds a_n[n] = x[n] * e
        @inbounds b_n[n] = c_e
        @inbounds b_n[M-n+2] = c_e
        @inbounds y[n] = e
    end

    A_X = zeros(T, M)

    A_X, B_X = fftr2!(A_X, a_n, e1, false)
    B_X, _ = fftr2!(B_X, b_n, e1, false)
    A_X .*= B_X
    B_X, A_X = fftr2!(B_X, A_X, e1, true)
    B_X .*= (1.0 / M)
    
    for i = 1:N
        @inbounds y[i] *= B_X[i]
    end
    y, x
end
