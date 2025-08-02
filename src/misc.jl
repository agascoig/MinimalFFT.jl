
# probably not useful

# uses Suppressor
macro write_native(filename, fn)
    return quote
        result = @capture_out @code_native $fn
        io = open($filename, "w")
        write(io, result)
        close(io)
    end
end

function bitrev!(x)
    N = length(x)
    shamt = leading_zeros(N) + 1
    for n = 0:N-1
        r = bitreverse(n)
        r = r >>> shamt
        if n < r
            @inbounds x[n+1], x[r+1] = x[r+1], x[n+1]
        end
    end
    x
end

function get_rv(T,N)
    x = zeros(Complex{T},N)
    o::Complex{T} = 1.0im

    for l=1:N
        x[l]=randn()+o*randn()
    end
    x
end


function dft_goertzel(x::Vector{T}, k) where {T<:Real}
    N = length(x)

    ω = 2.0*pi*k/N

    coeff = 2.0 * cos(ω)

    S_1::T = 0.0
    S_2::T = 0.0

    for n=1:N
        S_2, S_1 = S_1, x[n] + coeff*S_1 - S_2 
    end

    exp(1.0im*ω)*S_1 - S_2
end

function idft_duhamel(x::Vector{Complex{T}},X::Vector{Complex{T}}, forward_dft) where {T<:Real}
    N = length(X)
    Y = copy(X)
    # exchange real and imaginary parts of the initial sequence
    map!(a -> a.im+im*a.re, Y, Y)
    # perform a forward DFT
    forward_dft(x, Y)
    # exchange the real and imaginary parts of the result
    map!(a -> a.im/N+im*a.re/N, x, x)
end
