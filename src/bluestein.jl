
# Bluestein's FFT algorithm

function fft_bluestein!(y::AbstractVector{T}, x::AbstractVector{T}, 
    bp::Int64, stride::Int64, N::Int64, e1::Int64, inverse::Bool) where {T<:Complex}
    M = nextpow(2, 2 * N - 1)
    e1 = 63 - leading_zeros(M)

    a_n = zeros(T, M)
    b_n = zeros(T, M)

    impiN = inverse ? 1.0im * pi / N : -1.0im * pi / N

    @inbounds a_n[1] = x[bp]
    @inbounds b_n[1] = 1.0
    @inbounds y[bp] = 1.0
    for n = 2:N
        e = exp(impiN * (n-1) * (n-1))
        c_e = conj(e)
        @inbounds a_n[n] = x[bp+stride*(n-1)] * e
        @inbounds b_n[n] = c_e
        @inbounds b_n[M-n+2] = c_e
        @inbounds y[bp+stride*(n-1)] = e
    end

    A_X = zeros(T, M)

    A_X, B_X = fftr2!(A_X, a_n, 1, 1, M, e1, false)
    B_X, _ = fftr2!(B_X, b_n, 1, 1, M, e1, false)
    A_X .*= B_X
    B_X, A_X = fftr2!(B_X, A_X, 1, 1, M, e1, true)
    B_X .*= (1.0 / M)
    
    for i = 1:N
        @inbounds y[bp+stride*(i-1)] *= B_X[i]
    end
    y, x
end
