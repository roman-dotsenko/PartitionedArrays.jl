using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Testing Initial Guess Preservation ===")

# Test with the FEM Laplacian
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

# Test 1: Start with zero initial guess
println("Test 1: Zero initial guess")
y1 = zeros(size(A_fem,2))
println("Initial guess before: ", y1[1:3])
p1 = PS.linear_problem(y1, A_fem, b_fem)
s1 = PS.minres(p1; verbose=false, reltol=1e-12, iterations=1)  # Just 1 iteration
PS.solve(s1)
println("Solution after 1 iteration: ", PS.solution(p1)[1:3])
println()

# Test 2: Start with non-zero initial guess
println("Test 2: Non-zero initial guess")
y2 = ones(size(A_fem,2)) * 0.5  # Start with 0.5 everywhere
println("Initial guess before: ", y2[1:3])
p2 = PS.linear_problem(y2, A_fem, b_fem)
s2 = PS.minres(p2; verbose=false, reltol=1e-12, iterations=1)  # Just 1 iteration
PS.solve(s2)
println("Solution after 1 iteration: ", PS.solution(p2)[1:3])
println()

# The solutions should be different if initial guess is preserved!
println("Are the solutions the same? ", PS.solution(p1) ≈ PS.solution(p2))
