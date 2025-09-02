

import LinearAlgebra: transpose!

function reweight!(Y::AbstractVector{T}, L, M, inverse::Bool=false) where {T<:Complex}
    # L: num rows, M: num columns
    N = L * M
    @assert N == length(Y) "Lengths must be consistent length(Y)=$(length(Y)) N=$N=$L*$M."

    @inbounds begin
        W = one(T)
        W_step = one(T)
        C = inverse ? exp(2im * pi / N) : exp(-2im * pi / N)
        B = C

        l = 0
        for i = 1:N
            l += 1
            Y[i] *= W
            W *= W_step
            if l == L
                l = 0
                W = one(T)
                W_step = B
                B *= C
            end
        end
        Y
    end
end

function mixed_radix!(Y, X, e1, e2, N1, N2, fft1!, fft2!, inverse::Bool=false)
    N = N1 * N2

    @assert length(X) == length(Y) "Y and X must be same size"
    @assert N == length(X) "Incorrect rectangular decomposition, N=$N L=$L M=$M"

    @inbounds begin
        Y2D_LM = reshape(Y, (N1, N2))
        X2D_LM = reshape(X, (N1, N2))

        Y2D_LM, X2D_LM = do_fft(Y2D_LM, X2D_LM, fft2!, e2, 2, inverse)

        reweight!(reshape(Y2D_LM, N), N1, N2, inverse)

        X2D_LM, Y2D_LM = do_fft(X2D_LM, Y2D_LM, fft1!, e1, 1, inverse)

        Y2D_ML = reshape(Y2D_LM, (N2, N1))
        X2D_LM = reshape(X2D_LM, (N1, N2))

        transpose!(Y2D_ML, X2D_LM)

        Y = reshape(Y2D_ML, N)
        X = reshape(X2D_LM, N)

        Y, X
    end
end

function mixed_radix_weight_2_of_3(y3d::Array{T,S}, N, L, d, inverse::Bool=false) where {T<:Complex,S}
    # the weight is the same for all elements along d
    # L is inner dimension
    @inbounds begin
        dp = 0
        y = reshape(y3d, length(y3d))

        nd = ndims(y3d)
        soy = size(y3d)
        strides = [prod(soy[1:i-1]) for i = 1:nd]
        counts = zeros(Int64, nd)

        W_count = soy[d]

        W = one(T)
        W_step = one(T)
        C = inverse ? exp(2.0im * pi / N) : exp(-2.0im * pi / N)
        B = C

        stride = strides[d]
        l = 0
        dp = 1
        while true
            l += 1
            idx = dp
            for i = 1:W_count
                y[idx] *= W
                idx += stride
            end
            W *= W_step
            if l == L
                l = 0
                W = one(T)
                W_step = B
                B *= C
            end
            i = 1
            while i <= nd
                if i != d
                    counts[i] += 1
                    if counts[i] == soy[i]
                        counts[i] = 0
                        dp -= strides[i] * (soy[i] - 1)
                    else
                        dp += strides[i]
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
end

function mixed_radix!(Y, X, e1, e2, e3, N1, N2, N3, fft1!, fft2!, fft3!, inverse::Bool=false)
    @inbounds begin
        N = N1 * N2 * N3
        S123 = (N1, N2, N3)

        X123 = reshape(X, S123)
        Y123 = reshape(Y, S123)

        Y123, X123 = do_fft(Y123, X123, fft3!, e3, 3, inverse)

        mixed_radix_weight_2_of_3(Y123, N2 * N3, N2, 1, inverse)

        X123, Y123 = do_fft(X123, Y123, fft2!, e2, 2, inverse)

        mixed_radix_weight_2_of_3(X123, N1 * N2 * N3, N1, 2, inverse)
        mixed_radix_weight_2_of_3(X123, N1 * N2, N1, 3, inverse)

        Y123, X123 = do_fft(Y123, X123, fft1!, e1, 1, inverse)

        X321 = reshape(X123, (N3, N2, N1))

        permutedims!(X321, Y123, (3, 2, 1))

        Y = reshape(Y123, N)
        X = reshape(X123, N)

        X, Y
    end
end

