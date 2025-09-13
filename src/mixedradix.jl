

import LinearAlgebra: transpose!

function reweight!(Y::Vector{T}, N1::Int64, N2::Int64,
    bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
        N = N1 * N2
        @assert N == length(Y) "Lengths must be consistent length(Y)=$(length(Y)) N=$N=$N1*$N2."
        W = one(T)
        W_step = one(T)
        C = inverse ? exp(2im * pi / N) : exp(-2im * pi / N)
        B = C

        l = 0
        for i = 0:N-1
            l += 1
            Y[bp+i*instride] *= W
            W *= W_step
            if l == N1
                l = 0
                W = one(T)
                W_step = B
                B *= C
            end
        end
        Y
    end
end

function mixed_radix!(Y::Vector{T}, X::Vector{T}, e1::Int64, e2::Int64, N1::Int64, N2::Int64, fft1!, fft2!,
    bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
        N = N1 * N2
        Ns = (N1, N2)

        @assert length(X) == length(Y) "Y and X must be same size"
        @assert N == length(X) "Incorrect rectangular decomposition"

        Y2D_LM = reshape(Y, (N1, N2))
        X2D_LM = reshape(X, (N1, N2))

        Y2D_LM, X2D_LM = do_fft(Y2D_LM, X2D_LM, fft2!, Ns, e2, 2, bp, instride, inverse)

        reweight!(reshape(Y2D_LM, N), N1, N2, bp, instride, inverse)

        X2D_LM, Y2D_LM = do_fft(X2D_LM, Y2D_LM, fft1!, Ns, e1, 1, bp, instride, inverse)

        Y2D_ML = reshape(Y2D_LM, (N2, N1))
        X2D_LM = reshape(X2D_LM, (N1, N2))

        transpose!(Y2D_ML, X2D_LM)

        Y = reshape(Y2D_ML, N)
        X = reshape(X2D_LM, N)

        Y, X
    end
end

function mixed_radix_weight_2_of_3!(y3d::Array{T,S}, N::Int64, N1::Int64, d::Int64,
    Ns::Tuple{Vararg{Int64}}, bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex,S}
    # the weight is the same for all elements along d
    # L is inner dimension
    @inbounds begin
        y = reshape(y3d, length(y3d))
        nd = 3
        strides = [instride * prod(Ns[1:i-1]) for i = 1:nd]
        counts = zeros(Int64, nd)
        vlength = Ns[d]

        W = one(T)
        W_step = one(T)
        C = inverse ? exp(2.0im * pi / N) : exp(-2.0im * pi / N)
        B = C

        stride = strides[d]
        l = 0
        while true
            l += 1
            for i = 0:vlength-1
                y[bp+i*stride] *= W
            end
            W *= W_step
            if l == N1
                l = 0
                W = one(T)
                W_step = B
                B *= C
            end
            bp = indexer_count(d, nd, counts, strides, bp, Ns)
            if bp == 0
                break
            end
        end
    end
end

function mixed_radix!(Y::Vector{T}, X::Vector{T},
    e1::Int64, e2::Int64, e3::Int64,
    N1::Int64, N2::Int64, N3::Int64,
    fft1!, fft2!, fft3!,
    bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
        N = N1 * N2 * N3
        Ns = (N1, N2, N3)

        X123 = reshape(X, Ns)
        Y123 = reshape(Y, Ns)

        Y123, X123 = do_fft(Y123, X123, fft3!, Ns, e3, 3, bp, instride, inverse)

        mixed_radix_weight_2_of_3!(Y123, N2 * N3, N2, 1, Ns, bp, instride, inverse)

        X123, Y123 = do_fft(X123, Y123, fft2!, Ns, e2, 2, bp, instride, inverse)

        mixed_radix_weight_2_of_3!(X123, N1 * N2 * N3, N1, 2, Ns, bp, instride, inverse)
        mixed_radix_weight_2_of_3!(X123, N1 * N2, N1, 3, Ns, bp, instride, inverse)

        Y123, X123 = do_fft(Y123, X123, fft1!, Ns, e1, 1, bp, instride, inverse)

        X321 = reshape(X123, (N3, N2, N1))

        permutedims!(X321, Y123, (3, 2, 1))

        Y = reshape(Y123, N)
        X = reshape(X123, N)

        X, Y
    end
end
