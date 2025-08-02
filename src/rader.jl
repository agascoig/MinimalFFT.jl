
using Primes

function circular_convolution(NP::Int64, X1::Vector{T}, X2::Vector{T}) where {T<:Complex}
    # X1 and X2 are zero-padded from NP to L=2NP-1 length
    L = length(X1)

    X = zeros(T, L)

    e1 = 63 - leading_zeros(L)

    X1, X = fftr2!(X, X1, e1, false)
    X2, X = fftr2!(X, X2, e1, false)
    X2 .= X2 .* X1
    X, X2 = fftr2!(X, X2, e1, true)
    scale = 1.0 / L
    X .= scale * X

    X2 .= zeros(T, L)

    @inbounds X2[1:NP] = X[1:NP]

    @inbounds X2[1:NP-1] .+= X[NP+1:2NP-1] 
    X2, X1
end

function fft_rader!(y::Vector{T}, x::Vector{T}, e1::Int64, inverse::Bool=false) where {T<:Complex}
    N = length(x)
    L = nextpow(2, 2*N - 3)

    isprime(N) || error("Length must be prime")
    g = primitive_root(N)
    if g === nothing
        error("No primitive root found for $N")
    end

    indices = [powermod(g, q, N) for q in 0:N-2]
    neg_indices = [powermod(g, -q, N) for q in 0:N-2]

    X = zeros(T, L)
    c = inverse ? 2.0im * pi / N : -2.0im * pi / N
    @inbounds X[1:N-1] .= x[indices[1:N-1].+1]

    X2 = zeros(T, L)
    @inbounds X2[1:N-1] .= [exp(c * (neg_indices[q+1])) for q = 0:N-2]
        
    X, X2 = circular_convolution(N-1, X, X2)

    y[1] = sum(x)
    for k = 1:N-1
        @inbounds y[neg_indices[k]+1] = x[1] + X[k]
    end
    y, x
end

# There are primitive roots for non-primes, but no general algorithm to find them.
function primitive_root(n::Int)
    c = [1, 1, 2, 3, 2, 5, 3, 0, 2, 3, 2, 0, 2, 3, 0, 0, 3, 5, 2, 0, 0, 7, 5, 0, 2, 7, 2, 0, 2, 0, 3]

    if n < 1
        return nothing
    elseif n < length(c)
        return c[n] != 0 ? c[n] : nothing
    elseif isprime(n)
        phi_n = n - 1
        factors = Primes.factor(phi_n)

        for g in 2:n-1
            if gcd(g, n) == 1
                is_primitive = true
                for (p, _) in factors
                    if powermod(g, phi_n ÷ p, n) == 1
                        is_primitive = false
                        break
                    end
                end
                if is_primitive
                    return g
                end
            end
        end
    end
    nothing
end


