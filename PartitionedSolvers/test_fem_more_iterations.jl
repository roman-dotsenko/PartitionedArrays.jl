using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Testing FEM Laplacian with More Iterations ===")

# Create FEM Laplacian problem
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

println("Matrix size: ", size(A_fem))
println("True solution: ", x_true[1:5])

# Test MINRES with more iterations
y = zeros(size(A_fem,2))
p = PS.linear_problem(y, A_fem, b_fem)
s = PS.minres(p; verbose=false, reltol=1e-12, iterations=100)  # More iterations
PS.solve(s)

println("MINRES solution: ", PS.solution(p)[1:5])
println("MINRES error: ", norm(PS.solution(p) - x_true))

# Check residual
residual = A_fem * PS.solution(p) - b_fem
println("MINRES residual ||A*x - b||: ", norm(residual))
println("Relative residual: ", norm(residual) / norm(b_fem))
