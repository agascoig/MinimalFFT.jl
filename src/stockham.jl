
# 1-D Stockham autosorting routines

# References
# Takahashi, D. (2020). Fast Fourier Transformation Algorithms for Parallel Computers. Springer. 
# Nussbaumer, H.J. (1982). Fast Fourier Transform and Convolution Algorithms.  Springer.

function fftr2!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N ÷ 2
    m = 1

    u = 2 * pi / 2

    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(pmim * u / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    Y[bp+stride*(k+2*j*m)] = c0 + c1
                    Y[bp+stride*(k+2*j*m+m)] = w * (c0 - c1)
                end
                w = w * w_l
            end
            l >>>= 1
            m <<= 1
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr3!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N ÷ 3
    m = 1

    u = 2 * pi / 3

    c30 = 0.5
    c31 = sin(pi / 3)

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    c2 = X[bp+stride*(k+j*m+2*l*m)]
                    d0 = c1 + c2
                    d1 = c0 - c30 * d0
                    d2 = pmim * c31 * (c1 - c2)
                    Y[bp+stride*(k+3*j*m)] = c0 + d0
                    Y[bp+stride*(k+3*j*m+m)] = w * (d1 + d2)
                    Y[bp+stride*(k+3*j*m+2*m)] = w * w * (d1 - d2)
                end
                w = w * w_l
            end
            l ÷= 3
            m *= 3
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr4!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N >>> 2
    m = 1

    u = 2 * pi / 4

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+m*l)]
                    c2 = X[bp+stride*(k+j*m+2*m*l)]
                    c3 = X[bp+stride*(k+j*m+3*m*l)]
                    d0 = c0 + c2
                    d1 = c0 - c2
                    d2 = c1 + c3
                    d3 = c1 - c3
                    d3 = pmim * d3
                    Y[bp+stride*(k+4*j*m)] = d0 + d2
                    Y[bp+stride*(k+4*j*m+m)] = w * (d1 + d3)
                    Y[bp+stride*(k+4*j*m+2*m)] = w * w * (d0 - d2)
                    Y[bp+stride*(k+4*j*m+3*m)] = w * w * w * (d1 - d3)
                end
                w = w * w_l
            end
            l >>>= 2
            m <<= 2
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr5!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N ÷ 5
    m = 1

    u = 2 * pi / 5

    c50 = 0.25
    c51 = sin(2.0 * pi / 5.0)
    c52 = (sqrt(5) / 4)
    c53 = (sin(pi / 5.0) / sin(2.0 * pi / 5.0))

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    c2 = X[bp+stride*(k+j*m+2*l*m)]
                    c3 = X[bp+stride*(k+j*m+3*l*m)]
                    c4 = X[bp+stride*(k+j*m+4*l*m)]
                    d0 = c1 + c4
                    d1 = c2 + c3
                    d2 = c51 * (c1 - c4)
                    d3 = c51 * (c2 - c3)
                    d4 = d0 + d1
                    d5 = c52 * (d0 - d1)
                    d6 = c0 - c50 * d4
                    d7 = d6 + d5
                    d8 = d6 - d5
                    d9 = pmim * (d2 + c53 * d3)
                    d10 = pmim * (c53 * d2 - d3)
                    Y[bp+stride*(k+5*j*m)] = c0 + d4
                    Y[bp+stride*(k+5*j*m+m)] = w * (d7 + d9)
                    Y[bp+stride*(k+5*j*m+2*m)] = w * w * (d8 + d10)
                    Y[bp+stride*(k+5*j*m+3*m)] = (w^3) * (d8 - d10)
                    Y[bp+stride*(k+5*j*m+4*m)] = (w^4) * (d7 - d9)
                end
                w = w * w_l
            end
            l ÷= 5
            m *= 5
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr7!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N ÷ 7
    m = 1

    u = 2 * pi / 7

    c71 = -(cos(u) + cos(2u) + cos(3u)) / 3
    c72 = (2cos(u) - cos(2u) - cos(3u)) / 3
    c73 = (cos(u) - 2cos(2u) + cos(3u)) / 3
    c74 = (cos(u) + cos(2u) - 2cos(3u)) / 3
    c75 = (sin(u) + sin(2u) - sin(3u)) / 3
    c76 = (2sin(u) - sin(2u) + sin(3u)) / 3
    c77 = (-sin(u) + 2sin(2u) + sin(3u)) / 3
    c78 = (sin(u) + sin(2u) + 2sin(3u)) / 3

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    c2 = X[bp+stride*(k+j*m+2*l*m)]
                    c3 = X[bp+stride*(k+j*m+3*l*m)]
                    c4 = X[bp+stride*(k+j*m+4*l*m)]
                    c5 = X[bp+stride*(k+j*m+5*l*m)]
                    c6 = X[bp+stride*(k+j*m+6*l*m)]

                    a1 = c1 + c6
                    a2 = c1 - c6
                    a3 = c2 + c5
                    a4 = c2 - c5
                    a5 = c3 + c4
                    a6 = c3 - c4

                    a7 = a1 + a3 + a5
                    a8 = a1 - a5
                    a9 = -a3 + a5
                    a10 = -a1 + a3
                    a11 = a2 + a4 - a6
                    a12 = a2 + a6
                    a13 = -a4 - a6
                    a14 = -a2 + a4

                    m1 = c71 * a7
                    m2 = c72 * a8
                    m3 = c73 * a9
                    m4 = c74 * a10
                    m5 = -pmim * (c75 * a11)
                    m6 = -pmim * (c76 * a12)
                    m7 = -pmim * (c77 * a13)
                    m8 = -pmim * (c78 * a14)

                    x1 = c0 - m1
                    x2 = x1 + m2 + m3
                    x3 = x1 - m2 - m4
                    x4 = x1 - m3 + m4
                    x5 = m5 + m6 - m7
                    x6 = m5 - m6 - m8
                    x7 = -m5 - m7 - m8

                    Y[bp+stride*(k+7*j*m)] = c0 + a7
                    Y[bp+stride*(k+7*j*m+m)] = w * (x2 - x5)
                    Y[bp+stride*(k+7*j*m+2*m)] = w^2 * (x3 - x6)
                    Y[bp+stride*(k+7*j*m+3*m)] = w^3 * (x4 - x7)
                    Y[bp+stride*(k+7*j*m+4*m)] = w^4 * (x4 + x7)
                    Y[bp+stride*(k+7*j*m+5*m)] = w^5 * (x3 + x6)
                    Y[bp+stride*(k+7*j*m+6*m)] = w^6 * (x2 + x5)
                end
                w = w * w_l
            end
            l ÷= 7
            m *= 7
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr8!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64,inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N >>> 3
    m = 1

    u = 2 * pi / 8

    c81 = (sqrt(2) / 2)
    c82 = -(sqrt(2) / 2)

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    c2 = X[bp+stride*(k+j*m+2*l*m)]
                    c3 = X[bp+stride*(k+j*m+3*l*m)]
                    c4 = X[bp+stride*(k+j*m+4*l*m)]
                    c5 = X[bp+stride*(k+j*m+5*l*m)]
                    c6 = X[bp+stride*(k+j*m+6*l*m)]
                    c7 = X[bp+stride*(k+j*m+7*l*m)]
                    d0 = c0 + c4
                    d1 = c0 - c4
                    d2 = c2 + c6
                    d3 = pmim * (c2 - c6)
                    d4 = c1 + c5
                    d5 = c1 - c5
                    d6 = c3 + c7
                    d7 = c3 - c7
                    e0 = d0 + d2
                    e1 = d0 - d2
                    e2 = d4 + d6
                    e3 = pmim * (d4 - d6)
                    e4 = c81 * (d5 - d7)
                    e5 = -pmim * c82 * (d5 + d7)
                    e6 = d1 + e4
                    e7 = d1 - e4
                    e8 = d3 + e5
                    e9 = d3 - e5
                    Y[bp+stride*(k+8*j*m)] = e0 + e2
                    Y[bp+stride*(k+8*j*m+m)] = w * (e6 + e8)
                    Y[bp+stride*(k+8*j*m+2*m)] = w * w * (e1 + e3)
                    Y[bp+stride*(k+8*j*m+3*m)] = w^3 * (e7 - e9)
                    Y[bp+stride*(k+8*j*m+4*m)] = w^4 * (e0 - e2)
                    Y[bp+stride*(k+8*j*m+5*m)] = w^5 * (e7 + e9)
                    Y[bp+stride*(k+8*j*m+6*m)] = w^6 * (e1 - e3)
                    Y[bp+stride*(k+8*j*m+7*m)] = w^7 * (e6 - e8)
                end
                w = w * w_l
            end
            l ÷= 8
            m *= 8
            X, Y = Y, X
        end
        X, Y
    end
end

function fftr9!(Y::Vector{T}, X::Vector{T},
    N::Int64, e1::Int64, bp::Int64, stride::Int64, inverse::Bool) where {T<:Complex}
    @inbounds begin
    l = N ÷ 9
    m = 1

    u = 2 * pi / 9

    c90 = 0.5
    c91 = (3.0 / 2)
    c93 = (2cos(u) - cos(2u) - cos(4u)) / 3
    c94 = (cos(u) + cos(2u) - 2cos(4u)) / 3
    c95 = (cos(u) - 2cos(2u) + cos(4u)) / 3

    su = sin(u)
    s2u = sin(2u)
    s3u = sin(3u)
    s4u = sin(4u)

    r = inverse ? u : -u
    pmim = inverse ? 1.0im : -1.0im

        for t = 1:e1
            w_l = exp(im * r / l)
            w = one(T)
            for j = 0:l-1
                for k = 0:m-1
                    c0 = X[bp+stride*(k+j*m)]
                    c1 = X[bp+stride*(k+j*m+l*m)]
                    c2 = X[bp+stride*(k+j*m+2*l*m)]
                    c3 = X[bp+stride*(k+j*m+3*l*m)]
                    c4 = X[bp+stride*(k+j*m+4*l*m)]
                    c5 = X[bp+stride*(k+j*m+5*l*m)]
                    c6 = X[bp+stride*(k+j*m+6*l*m)]
                    c7 = X[bp+stride*(k+j*m+7*l*m)]
                    c8 = X[bp+stride*(k+j*m+8*l*m)]

                    t1 = c1 + c8
                    t2 = c2 + c7
                    t3 = c3 + c6
                    t4 = c4 + c5
                    t5 = t1 + t2 + t4
                    t6 = c1 - c8
                    t7 = c7 - c2
                    t8 = c3 - c6
                    t9 = c4 - c5
                    t10 = t6 + t7 + t9
                    t11 = t1 - t2
                    t12 = t2 - t4
                    t13 = t7 - t6
                    t14 = t7 - t9

                    m0 = c0 + t3 + t5
                    m1 = c91 * t3
                    m2 = -t5 * c90

                    t15 = -t12 - t11
                    m3 = c93 * t11
                    m4 = c94 * t12
                    m5 = c95 * t15

                    s0 = -m3 - m4
                    s1 = m5 - m4

                    m6 = pmim * s3u * t10
                    m7 = pmim * s3u * t8

                    t16 = -t13 + t14
                    m8 = -pmim * su * t13
                    m9 = -pmim * s4u * t14
                    m10 = -pmim * s2u * t16

                    s2 = -m8 - m9
                    s3 = m9 - m10
                    s4 = m0 + m2 + m2
                    s5 = s4 - m1
                    s6 = s4 + m2
                    s7 = s5 - s0
                    s8 = s1 + s5
                    s9 = s0 - s1 + s5
                    s10 = m7 - s2
                    s11 = m7 - s3
                    s12 = m7 + s2 + s3

                    Y[bp+stride*(k+9*j*m)] = m0
                    Y[bp+stride*(k+9*j*m+m)] = w * (s7 + s10)
                    Y[bp+stride*(k+9*j*m+2*m)] = w^2 * (s8 - s11)
                    Y[bp+stride*(k+9*j*m+3*m)] = w^3 * (s6 + m6)
                    Y[bp+stride*(k+9*j*m+4*m)] = w^4 * (s9 + s12)
                    Y[bp+stride*(k+9*j*m+5*m)] = w^5 * (s9 - s12)
                    Y[bp+stride*(k+9*j*m+6*m)] = w^6 * (s6 - m6)
                    Y[bp+stride*(k+9*j*m+7*m)] = w^7 * (s8 + s11)
                    Y[bp+stride*(k+9*j*m+8*m)] = w^8 * (s7 - s10)
                end
                w = w * w_l
            end
            l ÷= 9
            m *= 9
            X, Y = Y, X
        end
        X, Y
    end
end