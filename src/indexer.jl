
# indexer.jl - needed because column operator indexing for multi-dimensional array is slow

# do_fft_planned, do_1d, do_1d_r1 with P only for multi-dimensional FFTs

function do_fft_planned(P::MinimalPlan{T},
oy::Array{F,D}, ix::Array{F,E}, r::Int) where {T,F<:Number,D,E}
    if r==1
        oy, ix = do_1d_r1(P, oy, ix)
    else
        oy, ix = do_1d(P, oy, ix, r)
    end
    (oy, ix)
end

function do_1d(P::MinimalPlan{T}, oy::Array{F,D}, ix::Array{F,E},
r::Int64) where {T<:Number,F<:Number,D,E}
    soy = size(oy)
    nd = ndims(oy)
    bp = 1
    strides = [@inbounds prod(soy[1:i-1]) for i = 1:nd]
    counts = zeros(Int64, nd)
    vlength = soy[r]

    fft_y = Array{eltype(oy)}(undef, vlength)
    fft_x = Array{eltype(ix)}(undef, vlength)

    @inbounds stride = strides[r]
    while true
        idx = bp
        for i = eachindex(fft_x)
            @inbounds a = ix[idx]
            @inbounds fft_x[i] = a
            idx += stride
        end

        fft_y, fft_x = execute_plan(P, fft_y, fft_x, r)

        idx = bp
        for i = eachindex(fft_y)
            @inbounds a = fft_y[i]
            @inbounds oy[idx] = a
            idx += stride
        end

        i = 1
        while i <= nd
            if i != r
                @inbounds counts[i] += 1
                if @inbounds counts[i] == soy[i]
                    @inbounds counts[i] = 0
                    @inbounds bp -= strides[i] * (soy[i] - 1)
                else
                    @inbounds bp += strides[i]
                    break
                end
            end
            i += 1
        end
        if i > nd
            return (oy, ix)
        end
    end
end


function do_1d_r1(P::MinimalPlan{T}, oy::Array{F,D}, ix::Array{F,E}) where {T<:Number,F<:Number,D,E}
    vlength = size(oy,1)
    bp = 1
    limit = length(oy)
    flipped = false
    y = reshape(oy, length(oy))
    x = reshape(ix, length(ix))
    
    while bp < limit
        @inbounds fft_x = @view x[bp:bp + vlength-1]
        @inbounds fft_y = @view y[bp:bp + vlength-1]

        orig_fft_y = fft_y
        fft_y, fft_x = execute_plan(P, fft_y, fft_x, 1)

        if fft_y !== orig_fft_y
            flipped = true
        end
        bp += vlength
    end
    return flipped ? (ix, oy) : (oy, ix)
end

# do_1d, do_1d_r1, do_fft without P for decomposed FFTs

function do_1d(oy::Array{F,D}, ix::Array{F,E},
fn_name::Function, e1::Int64, r::Int64, inverse::Bool) where {F<:Number,D,E}
    soy = size(oy)
    nd = ndims(oy)
    bp = 1
    strides = [@inbounds prod(soy[1:i-1]) for i = 1:nd]
    counts = zeros(Int64, nd)
    vlength = soy[r]
    flipped = false
    y = reshape(oy, length(oy))
    x = reshape(ix, length(ix))
    orig_y = y

    @inbounds stride = strides[r]
    while true
        y, x = fn_name(y, x, bp, stride, vlength, e1, inverse)

        if y !== orig_y
            flipped = true
            y, x = x, y
        end

        i = 1
        while i <= nd
            if i != r
                @inbounds counts[i] += 1
                if @inbounds counts[i] == soy[i]
                    @inbounds counts[i] = 0
                    @inbounds bp -= strides[i] * (soy[i] - 1)
                else
                    @inbounds bp += strides[i]
                    break
                end
            end
            i += 1
        end
        if i > nd
            return flipped ? (ix, oy) : (oy, ix)
        end
    end
end

function do_1d_r1(oy::Array{F,D}, ix::Array{F,E},
fn_name::Function, e1::Int64, inverse::Bool) where {F<:Number,D,E}
    @inbounds vlength = size(oy,1)
    bp = 1
    limit = length(oy)
    flipped = false
    y = reshape(oy, length(oy))
    x = reshape(ix, length(ix))
    orig_y = y
    
    while bp < limit
        y, x = fn_name(y, x, bp, 1, vlength, e1, inverse)

        if y !== orig_y
            flipped = true
            y, x = x, y
        end
        bp += vlength
    end
    return flipped ? (ix, oy) : (oy, ix)
end

function do_fft(oy::Array{F,D}, ix::Array{F,E}, fn_name::Function,
e1::Int64, r::Int64, inverse::Bool) where {F<:Number,D,E}
    if r==1
        oy, ix = do_1d_r1(oy, ix, fn_name, e1, inverse)
    else
        oy, ix = do_1d(oy, ix, fn_name, e1, r, inverse)
    end
    (oy, ix)
end

