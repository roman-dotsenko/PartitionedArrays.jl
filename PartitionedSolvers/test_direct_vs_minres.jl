using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Comparing Direct vs MINRES for FEM Laplacian ===")

# Test with the FEM Laplacian
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

# Direct solver
println("Direct solver:")
x_direct = A_fem \ b_fem
error_direct = LinearAlgebra.norm(x_true - x_direct)
println("Direct solver error: ", error_direct)
println("Expected solution: ", x_true[1:5])
println("Direct solution: ", x_direct[1:5])
println()

# MINRES solver
println("MINRES solver:")
y_fem = zeros(size(A_fem,2))
p_fem = PS.linear_problem(y_fem, A_fem, b_fem)
s_fem = PS.minres(p_fem; verbose=false, reltol=1e-12, iterations=200)
PS.solve(s_fem)

error_minres = LinearAlgebra.norm(x_true - PS.solution(p_fem))
println("MINRES error: ", error_minres)
println("Expected solution: ", x_true[1:5])
println("MINRES solution: ", PS.solution(p_fem)[1:5])
println()

# Check matrix properties
println("Matrix properties:")
println("Matrix size: ", size(A_fem))
println("Matrix condition number: ", cond(Matrix(A_fem)))
eigenvals = eigvals(Matrix(A_fem))
println("Min eigenvalue: ", minimum(eigenvals))
println("Max eigenvalue: ", maximum(eigenvals))
