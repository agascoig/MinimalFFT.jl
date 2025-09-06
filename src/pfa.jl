#
# pfa.jl - Prime Factor Algorithm

# TBD: this has not been verified

import LinearAlgebra: transpose!

function extended_euclid(a::Int, b::Int)
    @assert a >= 0 && b >= 0 "a and b must be non-negative"
    if a == 0
        return (b, 0, 1)
    else
        (g, y, x) = extended_euclid(mod(b, a), a)
    end
    (g, x - (b ÷ a) * y, y)
end

function prime_factor!(Y::Vector{T}, X::Vector{T}, 
    e1::Int, e2::Int, N1::Int, N2::Int,
    fft1!::Function, fft2!::Function, 
    bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex}
    N = N1 * N2
    Ns = (N1, N2)

    (g, M1, M2) = extended_euclid(N1, N2)
    @assert g == 1 "prime_factor N1 and N2 must be coprime"

    # Not using standard Good-CRT mapping.  See "A Generalized Mixed-Radix Algorithm for
    # Memory-Based FFT Processors."  IEEE Tran. on Circuits and Systems-II.  Jan 2010.

    mask_mux_mod(a, B) = a - (B & -(a ≥ B))

    Q1P = mod(M2, N1)

    rhs_n = 0
    L2 = 0
    for n1p = 0:N1-1
        R1 = 0
        L2 = 0
        for n2p = 0:N2-1
            n1 = mask_mux_mod(n1p + R1, N1)
            lhs_n = n1 + L2
            @inbounds Y[bp+instride*lhs_n] = X[bp+instride*rhs_n]
            R1 = mask_mux_mod(R1 + Q1P, N1)
            rhs_n += 1
            L2 += N1
        end
    end

    Y2D_N1N2 = reshape(Y, (N1, N2))
    X2D_N1N2 = reshape(X, (N1, N2))

    X2D_N1N2, Y2D_N1N2 = do_fft(X2D_N1N2, Y2D_N1N2, fft1!, Ns, e1, 1, bp, instride, inverse)
    Y2D_N1N2, X2D_N1N2 = do_fft(Y2D_N1N2, X2D_N1N2, fft2!, Ns, e2, 2, bp, instride, inverse)

    Y = reshape(Y2D_N1N2, N)
    X = reshape(X2D_N1N2, N)
    
    Q2P = mod(M1, N2)

    lhs_k = 0
    for k2p = 0:N2-1
        R1 = 0
        for k1p = 0:N1-1
            k2 = mask_mux_mod(k2p + R1, N2)
            rhs_k = k1p + k2 * N1
            @inbounds X[bp+instride*lhs_k] = Y[bp+instride*rhs_k]
            R1 = mask_mux_mod(R1 + Q2P, N2)
            lhs_k += 1
        end
    end
    (X, Y)
end

function Qs(N1::Int, N2::Int, N3::Int)
    (g1, p1, q1) = extended_euclid(N1, N2 * N3)
    (g2, p2, q2) = extended_euclid(N2, N1 * N3)
    (g3, p3, q3) = extended_euclid(N3, N1 * N2)
    (g4, p4, q4) = extended_euclid(N2 * N3, N1)

    @assert g1 == 1 && g2 == 1 && g3 == 1 && g4 == 1 "N1, N2, N3 must be coprime"
    (p1, p2, p3, p4, -q1, -q2 * N1, -q3 * N1, -q4)
end

function nmap!(Y, X, bp, instride, N1, N2, N3, Q1P, Q2P)
    mask_mux_mod(a, B) = a - (B & -(a ≥ B))

    rhs_n = 0
    for n1p = 0:N1-1
        R1 = 0
        for n2p = 0:N2-1
            R2 = 0
            for n3p = 0:N3-1
                n1 = mask_mux_mod(n1p + R1, N1)
                n2 = mask_mux_mod(n2p + R2, N2)
                lhs_n = n1 + N1 * n2 + N1 * N2 * n3p
                @inbounds Y[bp+instride*lhs_n] = X[bp+instride*rhs_n]
                R1 = mask_mux_mod(R1 + Q1P, N1)
                R2 = mask_mux_mod(R2 + Q2P, N2)
                rhs_n += 1
            end
        end
    end
end

function kmap!(Y, X, bp, instride, N1, N2, N3, P1, P2)
    mask_mux_mod(a, B) = a - (B & -(a ≥ B))

    lhs_k = 0
    for k3p = 0:N3-1
        R2 = 0
        for k2p = 0:N2-1
            R1 = 0
            for k1p = 0:N1-1
                k2 = mask_mux_mod(k2p + R1, N2)
                k3 = mask_mux_mod(k3p + R2, N3)
                rhs_k = k1p + N1 * k2 + N1 * N2 * k3
                @inbounds Y[bp+instride*lhs_k] = X[bp+instride*rhs_k]
                R1 = mask_mux_mod(R1 + P1, N2)
                R2 = mask_mux_mod(R2 + P2, N3)
                lhs_k += 1
            end
        end
    end
end

function prime_factor!(Y::Vector{T}, X::Vector{T}, e1::Int64, e2::Int64, e3::Int64, 
    N1::Int64, N2::Int64, N3::Int64, # embedded sizes
    fft1!::Function, fft2!::Function, fft3!::Function, 
    bp::Int64, instride::Int64, inverse::Bool) where {T<:Complex}
    N = N1 * N2 * N3
    Ns = (N1, N2, N3)

    B = (p1, p2, p3, p4, Q1, Q2, Q3, Q4) = Qs(N1, N2, N3)

    Q1P = mod(-Q1, N1)
    Q2P = mod(-Q2, N2)

    P1 = mod(-Q4, N2)
    P2 = mod(-Q3 ÷ N1, N3)

    nmap!(Y, X, bp, instride, N1, N2, N3, Q1P, Q2P)

    S123 = (N1, N2, N3)

    Y123 = reshape(Y, S123)
    X123 = reshape(X, S123)

    X123, Y123 = do_fft(X123, Y123, fft1!, Ns, e1, 1, bp, instride, inverse)
    Y123, X123 = do_fft(Y123, X123, fft2!, Ns, e2, 2, bp, instride, inverse)
    X123, Y123 = do_fft(X123, Y123, fft3!, Ns, e3, 3, bp, instride, inverse)

    Y = reshape(Y123, N)
    X = reshape(X123, N)

    kmap!(Y, X, bp, instride, N1, N2, N3, P1, P2)
    (Y, X)
end
