using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Checking MINRES Solution Residual ===")

# Test with the FEM Laplacian
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

# Get MINRES solution
y_fem = zeros(size(A_fem,2))
p_fem = PS.linear_problem(y_fem, A_fem, b_fem)
s_fem = PS.minres(p_fem; verbose=false, reltol=1e-12, iterations=200)
PS.solve(s_fem)

x_minres = PS.solution(p_fem)

println("True solution: ", x_true[1:5])
println("MINRES solution: ", x_minres[1:5])
println()

# Check residuals
residual_minres = A_fem * x_minres - b_fem
println("MINRES residual ||A*x_minres - b||: ", norm(residual_minres))
println("MINRES residual[1:5]: ", residual_minres[1:5])
println()

# Check what MINRES actually solved
b_actual = A_fem * x_minres
println("MINRES actually solved for b_actual = A*x_minres:")
println("b_actual[1:5]: ", b_actual[1:5])
println("Original b[1:5]: ", b_fem[1:5])
println("Difference (b - b_actual)[1:5]: ", (b_fem - b_actual)[1:5])
println()

# Let's see if MINRES solved the wrong problem
println("Checking if MINRES solved A*x = 0 instead of A*x = b:")
println("||A*x_minres||: ", norm(A_fem * x_minres))
println("||b||: ", norm(b_fem))

# Check scaling
scale_factor = norm(b_fem) / norm(A_fem * x_minres)
println("Scale factor b/A*x_minres: ", scale_factor)
println("x_minres * scale_factor[1:5]: ", (x_minres * scale_factor)[1:5])
