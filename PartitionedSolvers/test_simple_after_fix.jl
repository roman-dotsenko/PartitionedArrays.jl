using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Testing Simple 2*I after Initial Guess Fix ===")

# Test with the simple 2*I matrix
n = 5
A_simple = 2 * I(n)
x_true = ones(n)
b_simple = A_simple * x_true

println("Matrix A:")
display(Matrix(A_simple))
println("RHS b: ", b_simple)
println("True solution: ", x_true)

# Test MINRES
y = zeros(n)
p = PS.linear_problem(y, A_simple, b_simple)
s = PS.minres(p; verbose=false, reltol=1e-12, iterations=20)
PS.solve(s)

println("MINRES solution: ", PS.solution(p))
println("MINRES error: ", norm(PS.solution(p) - x_true))

# Check residual
residual = A_simple * PS.solution(p) - b_simple
println("MINRES residual ||A*x - b||: ", norm(residual))
