
using Nemo

# Step 1: Coefficient field: Q(W₁, x₁)
R, (W_1, x_1) = rational_function_field(QQ, ["W_1", "x_1"])

# Step 2: Polynomial ring in x over R
S, x = polynomial_ring(R, "x")

a = x^2 * W_1 * x_1

b = x^2 + x + 1

println("a: $a")
println("typeof(a): $(typeof(a))")
println("b: $b")

r = rem(a, b)
println("r: $r")
