
# indexer.jl - needed because column operator indexing for multi-dimensional array is slow

# do_fft_planned, do_1d, do_1d_r1 with P only for multi-dimensional FFTs
# (not due to decomposition)

function do_fft_planned(P::MinimalPlan{T},
    oy::Array{F,D}, ix::Array{F,D}, r::Int64) where {T,F<:Complex,D}
    if r == 1
        oy, ix = do_1d_r1(P, oy, ix)
    else
        oy, ix = do_1d(P, oy, ix, r)
    end
    (oy, ix)
end

function do_1d(P::MinimalPlan{T}, oy::Array{F,D}, ix::Array{F,D},
    r::Int64) where {T<:Number,F<:Complex,D}
    @inbounds begin
        Ns = size(oy)
        nd = ndims(oy)
        strides = [prod(Ns[1:i-1]) for i = 1:nd]
        counts = zeros(Int64, nd)
        bp = 1
        flipped = false

        y = reshape(oy, length(oy))
        x = reshape(ix, length(ix))

        orig_y = y
        stride = strides[r]
        while true
            y, x = execute_plan(P, y, x, r, bp, stride)
            if y !== orig_y
                flipped = true
                y, x = x, y
            end
            bp = indexer_count(r, nd, counts, strides, bp, Ns)
            if bp == 0
                return flipped ? (ix, oy) : (oy, ix)
            end
        end
    end
end


function do_1d_r1(P::MinimalPlan{T}, oy::Array{F,D}, ix::Array{F,D}) where {T<:Number,F<:Complex,D}
    vlength = size(oy, 1)
    bp = 1
    limit = prod(size(oy))
    flipped = false
    y = reshape(oy, length(oy))
    x = reshape(ix, length(ix))

    orig_y = y
    while bp < limit
        y, x = execute_plan(P, y, x, 1, bp, 1)

        if y !== orig_y
            flipped = true
        end
        bp += vlength
    end
    return flipped ? (ix, oy) : (oy, ix)
end

# do_1d, do_1d_r1, do_fft without P for decomposed FFTs with
# dimensions Ns.  These may or may not be embedded in a larger
# multi-dimensional FFT.

function do_1d(oy::Array{F,D}, ix::Array{F,D},
    fn_name::Function, Ns::Tuple{Vararg{Int64}},
    e1::Int64, r::Int64, bp::Int64, instride::Int64, inverse::Bool) where {F<:Complex,D}
    @inbounds begin
        nd = length(Ns)
        strides = [instride * prod(Ns[1:i-1]) for i = 1:nd]
        counts = zeros(Int64, nd)
        vlength = Ns[r]
        flipped = false
        y = reshape(oy, length(oy))
        x = reshape(ix, length(ix))
        orig_y = y

        stride = strides[r]
        while true
            y, x = fn_name(y, x, vlength, e1, bp, stride, inverse)

            if y !== orig_y
                flipped = true
                y, x = x, y
            end
            bp = indexer_count(r, nd, counts, strides, bp, Ns)
            if bp == 0
                return flipped ? (ix, oy) : (oy, ix)
            end
        end
    end
end

function do_1d_r1(oy::Array{F,D}, ix::Array{F,D},
    fn_name::Function, Ns::Tuple{Vararg{Int64}},
    e1::Int64, bp::Int64, instride::Int64, inverse::Bool) where {F<:Complex,D}
    @inbounds begin
        vlength = Ns[1]
        flipped = false
        y = reshape(oy, length(oy))
        x = reshape(ix, length(ix))
        orig_y = y
        limit = prod(Ns[2:end])
        l = 0

        while l < limit
            l += 1
            y, x = fn_name(y, x, vlength, e1, bp, instride, inverse)

            if y !== orig_y
                flipped = true
                y, x = x, y
            end
            bp += instride * vlength
        end
        return flipped ? (ix, oy) : (oy, ix)
    end
end

function do_fft(oy::Array{F,D}, ix::Array{F,D}, fn_name::Function,
    Ns::Tuple{Vararg{Int64}}, # embedded size of decomposed FFT
    e1::Int64, r::Int64, bp::Int64, instride::Int64, inverse::Bool) where {F<:Complex,D}
    if r == 1
        oy, ix = do_1d_r1(oy, ix, fn_name, Ns, e1, bp, instride, inverse)
    else
        oy, ix = do_1d(oy, ix, fn_name, Ns, e1, r, bp, instride, inverse)
    end
    (oy, ix)
end

# update indexer counters
function indexer_count(r::Int64, nd::Int64, counts::Vector{Int64},
    strides::Vector{Int64}, bp::Int64, Ns::Tuple{Vararg{Int64}})
    @inbounds begin
        i = 1
        while i <= nd
            if i != r
                counts[i] += 1
                if counts[i] == Ns[i]
                    counts[i] = 0
                    bp -= strides[i] * (Ns[i] - 1)
                else
                    bp += strides[i]
                    break
                end
            end
            i += 1
        end
        return (i > nd) ? 0 : bp
    end
end

