
# Winograd Fourier Transform Algorithm (WFTA)

using Nemo, Symbolics, Primes

# Wang, Angie.  Ph.D. Dissertation, UC Berkeley.
# "Agile Design of Generator-Based Signal Processing
# Hardware," 2018. p.44-51

macro write_text(x)
    return quote
        println(string($(QuoteNode(x))),"=")
        show(stdout, "text/plain", $(esc(x)))
        println("\n")
    end
end

function direct_dft(x, W::Vector{S}, X::Vector{T}, e1::Int64, inverse::Bool) where {S,T}
    N = length(X)
    y = Vector{typeof(x)}(undef, N)
    for k = 1:N
        y[k] = 0*x
        for n = 1:N
            y[k] += (X[n] * W[(n-1)*(k-1)%N+1])
        end
    end
    y
end

function primitive_root(n::Int)
    c = Dict(1=>1,2=>1,4=>3,5=>2,6=>5,7=>3,9=>2,
    10=>3,11=>2,13=>2,14=>3,17=>3,18=>5,19=>2,
    22=>7,23=>5,25=>2,26=>7,29=>2,31=>3)
    
    if haskey(c, n)
        return c[n]
    end
    if isprime(n)
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
    return nothing  # No primitive root exists
end

function cyclo_poly(n, FF, x)
    R, z = polynomial_ring(ZZ, "z")
    f = cyclotomic(n, z)

    f2 = zero(FF)
    d = degree(f)
    for i=0:d
        f2 += coeff(f,i)*x^i
    end
    f2
end

function cyclotomic_irreducible(n, FF, x)
# cannot call this due to ZZRingElem being required
#    [Nemo.cyclotomic(d, x) for d in Primes.divisors(n)]
    [cyclo_poly(d, FF, x) for d in Primes.divisors(n)]
end

function gen_poly(x, FF, A, B)
    m = size(A, 1)

    ipoly = cyclotomic_irreducible(m, FF, x)

    k_num = length(ipoly)
    c = Dict{Tuple{Int,Int},eltype(ipoly)}()

    # Precompute c_ij coefficients for polynomials
    for i in 1:k_num-1
        for j in i+1:k_num
            c[(i, j)] = invmod(ipoly[i], ipoly[j])  # (2.163) Find inverse of m_i mod m_j
        end
    end

    Am = sum([A[m-j] * x^j for j = 0:m-1]) # (2.149)
    Bm = sum([B[j+1] * x^j for j = 0:m-1]) # (2.150)
    uprod = Am * Bm
    uprod = expand(uprod)

    TT = Vector{eltype(ipoly)}
    u = TT(undef, k_num)

    for i = 1:k_num
        mpoly = ipoly[i]
        u[i] = rem(uprod, mpoly) # (2.162+)
    end
    u = simplify(u)

    v = TT(undef, k_num)
    v[1] = u[1]
    for i = 2:k_num
        v[i] = u[i]
        for j = 1:i-1
            v[i] = rem((v[i] - v[j]) * c[(j, i)], ipoly[i]) # (2.167)
        end
    end

    P = v[1]
    for i = 1:k_num-1
        P = P + v[i+1] * prod(ipoly[1:i]) # (2.168)
    end
    P
end

function subscript(index)
    subscripts = "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089"
    digits = string(index)
    sub_str = join([subscripts[parse(Int, d)*3+1] for d in digits])
    return sub_str
end

function WFTA(N)
    g = primitive_root(N)
    if g === nothing
        error("No primitive root found for N=$N")
    end
    gs_A = [powermod(g, i, N) for i in 2:N]
    gs_B = [powermod(g, i, N) for i in 1:N-1]

    K, zeta = cyclotomic_field(N, "W")

    W = [zeta^k for k in 0:N-1]

    x_names = ["x$(subscript(i))" for i in 0:N-1]

    us_names = ["u$(subscript(i))" for i in 1:N-1]
    vs_names = ["v$(subscript(i))" for i in 1:N-1]

    all_names = vcat(x_names, us_names, vs_names)

    S1, all_vars = rational_function_field(K, all_names)
    A = [W[gs_A[i]+1] for i in 1:N-1]
    B = [all_vars[gs_B[i]+1] for i in 1:N-1]
    us = [all_vars[i+N] for i in 1:N-1]
    vs = [all_vars[i+2N-1] for i in 1:N-1]

    S2, x = polynomial_ring(S1, "x")

    P = gen_poly(x, S2, A, B) # (2.147)

    X = Vector{eltype(P)}(undef, N)
    X[1] = sum([all_vars[i] for i = 1:N])
    for i = 1:N-1
        X[gs_B[i]+1] = all_vars[1] + coeff(P, (N-1)-i)
    end

# commented out, for testing:
#    DX = direct_dft(x, [zeta^k for k in 0:N-1],
#    [all_vars[i] for i in 1:N],0,false)
#    X, DX
    X
end


