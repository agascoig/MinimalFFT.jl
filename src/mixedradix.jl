

import LinearAlgebra: transpose!

function reweight!(Y::Vector{T}, L, M, inverse::Bool=false) where {T<:Complex}
    # L: num rows, M: num columns
    N = L * M
    @assert N == length(Y) "Lengths must be consistent length(Y)=$(length(Y)) N=$N=$L*$M."

    W = 1.0 + 0.0im
    W_step = 1.0 + 0.0im
    C = inverse ? exp(2im * pi / N) : exp(-2im * pi / N)
    B = C

    l = 0
    for i = 1:N
        l += 1
        @inbounds Y[i] *= W
        W *= W_step
        if l == L
            l = 0
            W = 1.0 + 0.0im
            W_step = B
            B *= C
        end
    end
    Y
end

function mixed_radix!(Y, X, e1, e2, N1, N2, fft1!, fft2!, inverse::Bool=false)
    # L*M matrix
    N = N1 * N2

    @assert length(X) == length(Y) "Y and X must be same size"
    @assert N == length(X) "Incorrect rectangular decomposition, N=$N L=$L M=$M"

    Y2D_ML = reshape(Y, (N2, N1))
    X2D_ML = reshape(X, (N2, N1))
    Y2D_LM = reshape(Y, (N1, N2))
    X2D_LM = reshape(X, (N1, N2))

    do_1d(Y2D_LM, X2D_LM, e2, 2, fft2!, inverse)

    reweight!(Y, N1, N2, inverse)

    do_1d(X2D_LM, Y2D_LM, e1, 1, fft1!, inverse)

    transpose!(Y2D_ML, X2D_LM)
    Y, X
end

function mixed_radix_weight_2_of_3(y3d, y, N, L, d, inverse::Bool=false)
    # the weight is the same for all elements along d
    # L is inner dimension
    dp = 0

    nd = ndims(y3d)
    soy = size(y3d)
    strides = [prod(soy[1:i-1]) for i = 1:nd]
    counts = zeros(Int64, nd)

    W_count = soy[d]

    W = 1.0 + 0.0im
    W_step = 1.0 + 0.0im
    C = inverse ? exp(2.0im * pi / N) : exp(-2.0im * pi / N)
    B = C

    stride = strides[d]
    l = 0
    dp = 1
    while true
        l += 1
        idx = dp
        for i = 1:W_count
            @inbounds y[idx] *= W
            idx += stride
        end
        W *= W_step
        if l == L
            l = 0
            W = 1.0 + 0.0im
            W_step = B
            B *= C
        end
        i = 1
        while i <= nd
            if i != d
                @inbounds counts[i] += 1
                if @inbounds counts[i] == soy[i]
                    @inbounds counts[i] = 0
                    @inbounds dp -= strides[i] * (soy[i] - 1)
                else
                    @inbounds dp += strides[i]
                    break
                end
            end
            i += 1
        end
        if i > nd
            break
        end
    end
end

function mixed_radix!(Y, X, e1, e2, e3, N1, N2, N3, fft1!, fft2!, fft3!, inverse::Bool=false)
    S123 = (N1, N2, N3)

    X123 = reshape(X, S123)
    Y123 = reshape(Y, S123)

    do_1d(Y123, X123, e3, 3, fft3!, inverse)

    mixed_radix_weight_2_of_3(Y123, Y, N2 * N3, N2, 1, inverse)

    do_1d(X123, Y123, e2, 2, fft2!, inverse)

    mixed_radix_weight_2_of_3(X123, X, N1 * N2 * N3, N1, 2, inverse)
    mixed_radix_weight_2_of_3(X123, X, N1 * N2, N1, 3, inverse)

    do_1d(Y123, X123, e1, 1, fft1!, inverse)

    X321 = reshape(X, (N3, N2, N1))
    permutedims!(X321, Y123, (3, 2, 1))
    X, Y
end

