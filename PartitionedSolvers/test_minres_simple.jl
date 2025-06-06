using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

# Simple test case without preconditioner
tol = 1e-7

# Create a simple positive definite matrix
n = 10
A_simple = Matrix{Float64}(I, n, n) * 2.0  # 2*I matrix
x_true = ones(n)
b_simple = A_simple * x_true
y_simple = zeros(n)

println("=== Test 1: Simple 2*I matrix (no preconditioner) ===")
p_simple = PS.linear_problem(y_simple, A_simple, b_simple)
s_simple = PS.minres(p_simple; verbose=true, reltol=tol)
PS.solve(s_simple)

error_simple = LinearAlgebra.norm(x_true - PS.solution(p_simple))
println("Solution error (no preconditioner): ", error_simple)
println("Expected solution: ", x_true[1:5])
println("Actual solution: ", PS.solution(p_simple)[1:5])
println()

# Test with the FEM Laplacian but no preconditioner
println("=== Test 2: FEM Laplacian (no preconditioner) ===")
n2 = 5  # Smaller for easier debugging
args = PA.laplacian_fem((n2,n2,n2))
A_fem = PA.sparse_matrix(args...)
x_fem_true = ones(size(A_fem,2))
b_fem = A_fem * x_fem_true
y_fem = zeros(size(A_fem,2))

p_fem = PS.linear_problem(y_fem, A_fem, b_fem)
s_fem = PS.minres(p_fem; verbose=true, reltol=tol)
PS.solve(s_fem)

error_fem = LinearAlgebra.norm(x_fem_true - PS.solution(p_fem))
println("Solution error (FEM, no preconditioner): ", error_fem)
println("Expected solution: ", x_fem_true[1:5])
println("Actual solution: ", PS.solution(p_fem)[1:5])
