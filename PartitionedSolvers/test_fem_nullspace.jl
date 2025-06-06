using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

# Check if FEM matrix has null space issues
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)

println("Checking matrix properties...")
A_dense = Matrix(A_fem)
eigenvalues = eigvals(A_dense)
min_eigval = minimum(eigenvalues)
println("Minimum eigenvalue: ", min_eigval)

# Check if constant vector is in null space (common for Laplacian)
ones_vec = ones(size(A_fem,2))
A_ones = A_fem * ones_vec
println("||A * ones|| = ", norm(A_ones))
println("This should be > 0 if ones is not in null space")

# Check what the direct solver actually gives for A*x = A*ones
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true
x_direct = A_dense \ b_fem

println("True solution: ", x_true[1:10])
println("Direct solution: ", x_direct[1:10])
println("Difference: ", (x_true - x_direct)[1:10])

# Also check with a random RHS to see if the issue is specific to b = A*ones
println("\n=== Test with random RHS ===")
x_random_true = randn(size(A_fem,2))
b_random = A_fem * x_random_true
x_random_direct = A_dense \ b_random
println("Random solution error (direct): ", norm(x_random_true - x_random_direct))

# Test MINRES with random RHS
y_random = zeros(size(A_fem,2))
p_random = PS.linear_problem(y_random, A_fem, b_random)
s_random = PS.minres(p_random; verbose=false, reltol=1e-10)
PS.solve(s_random)
println("Random solution error (MINRES): ", norm(x_random_true - PS.solution(p_random)))
