
# indexer.jl - needed because column operator indexing for multi-dimensional array is slow

function do_fft_planned(P::MyPlan{T},
oy::Array{F,D}, ix::Array{F,E}, r::Int) where {T,F<:Number,D,E}
    soy = size(oy)
    len = soy[r]
    fft_y = Array{eltype(oy)}(undef, len)
    fft_x = Array{eltype(ix)}(undef, len)
    do_fft_planned(P, oy, ix, fft_y, fft_x, execute_plan, 0, r)
    oy, ix
end

function do_fft_planned(P::Union{MyPlan{T}}, oy::Array{F,D}, ix::Array{F,E},
fft_y::Vector{C}, fft_x::Vector{C}, fn_name::Function, e1::Int64,
r::Int64) where {T,C<:Complex,F<:Number,D,E}

    soy = size(oy)
    nd = ndims(oy)
    dp = 1
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

        fft_y, fft_x = fn_name(P, fft_y, fft_x, r)

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

function do_1d(oy::Array{F,D}, ix::Array{F,E}, fft_y::Vector{C}, 
fft_x::Vector{C}, fn_name::Function, e1::Int64, r::Int64,
inverse::Bool) where {C<:Complex,F<:Number,D,E}

    soy = size(oy)
    nd = ndims(oy)
    dp = 1
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

function do_fft(oy::Array{F,D}, ix::Array{F,E}, fn_name::Function,
e1::Int64, r::Int64, inverse::Bool) where {F<:Number,D,E}
    soy = size(oy)
    len = soy[r]
    fft_y = Array{eltype(oy)}(undef, len)
    fft_x = Array{eltype(ix)}(undef, len)
    do_1d(oy, ix, fft_y, fft_x, fn_name, e1, r, inverse)
    oy, ix
end

