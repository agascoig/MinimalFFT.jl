
# Redefine @inbounds
@eval macro inbounds(ex)
    return esc(ex)  # ignore inbounds and keep checks
end

function print_dft(x, name)
    println("\nreal:")
    for k=1:length(x)
        println("$name[$k] = ",real(x[k]))
    end
    println("\nimag:")
    for k=1:length(x)
        println("$name[$k] = ",imag(x[k]))
    end
end

using Infiltrator, FFTW, Symbolics

include("../src/stockham.jl")
include("../src/rader.jl")
include("../src/direct.jl")

N = 11
@variables bad::Complex, x[0:N-1]::Complex
x_direct = collect(x)
x_rader = collect(x)
y_direct = [bad for i = 1:N]
y_rader = [bad for i = 1:N]

function remove_small_terms(expr, tol=1e-12)

    function walk_remove(e, comp, tol=1e-12)
        if !(hasproperty(comp, :val) && hasproperty(comp.val, :dict))
            return e
        end
        d = copy(comp.val.dict)
        e = comp
        for (key, value) in d
            if value isa Number && abs(value) < tol
                e = substitute(e, Dict(key => 0))
            end
        end
        e
    end

    # Create a rule to remove small coefficients
   rule = @rule x => begin
        if !isa(x, Complex)
            return x
        end
        (abs(real(x)) < tol && abs(imag(x))) < tol ? 0 : x
    end

    expr = expand(expr)
    expr = simplify(expr, rule)
    r = walk_remove(expr, expr.re)
    i = walk_remove(expr, expr.im)
    simplify(r + im * i)
end

y_direct, x_direct = direct_dft!(y_direct, x_direct, false)
y_rader, x_rader = fft_rader!(y_rader, x_rader, false)

y_rader = remove_small_terms.(y_rader);
y_direct = remove_small_terms.(y_direct);

diff = remove_small_terms.(y_rader .- y_direct)

println(diff)

nothing

#print_dft(y_direct, "direct")
#print_dft(y_rader, "rader")
