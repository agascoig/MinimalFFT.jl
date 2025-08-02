# indexer.jl - needed because column operator indexing for multi-dimensional array is slow

function do_1d_inner(oy, ix, fft_y, fft_x, e1::Int64, r::Int64, fn_name, inverse::Bool)
    nd = ndims(oy)
    dp = 1
    soy = size(oy)
    strides = [@inbounds prod(soy[1:i-1]) for i = 1:nd]
    counts = zeros(Int64, nd)

    @inbounds stride = strides[r]
    while true
        idx = dp
        for i = eachindex(fft_x)
            @inbounds a = ix[idx]
            @inbounds fft_x[i] = a
            idx += stride
        end
        fft_y, fft_x = fn_name(fft_y, fft_x, e1, inverse)
        idx = dp
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
                    @inbounds dp -= strides[i] * (soy[i] - 1)
                else
                    @inbounds dp += strides[i]
                    break
                end
            end
            i += 1
        end
        if i > nd
            return
        end
    end
end

function do_1d(oy, ix, e1, r, fn_name, inverse::Bool)
    len = size(oy)[r]
    fft_y = Array{eltype(oy)}(undef, len)
    fft_x = Array{eltype(ix)}(undef, len)

    do_1d_inner(oy, ix, fft_y, fft_x, e1, r, fn_name, inverse)
end

