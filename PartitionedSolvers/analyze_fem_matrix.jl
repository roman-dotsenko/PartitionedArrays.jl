using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

println("=== Analyzing FEM Laplacian Matrix Properties ===")

# Test with the FEM Laplacian
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

println("Matrix properties:")
println("Matrix size: ", size(A_fem))

# Check if the matrix has a non-trivial null space
println("\nNull space analysis:")
eigenvals = eigvals(Matrix(A_fem))
min_eig = minimum(eigenvals)
max_eig = maximum(eigenvals)
println("Min eigenvalue: ", min_eig)
println("Max eigenvalue: ", max_eig)
println("Matrix condition number: ", max_eig/min_eig)

# Check if the matrix is singular or near-singular
println("Rank of matrix: ", rank(Matrix(A_fem)))
println("Matrix size: ", size(A_fem, 1))

# Check if b is in the range of A
println("\nRight-hand side analysis:")
println("||b||: ", norm(b_fem))
println("||A*ones||: ", norm(A_fem * ones(size(A_fem,2))))

# Check if ones is in the null space
null_residual = A_fem * ones(size(A_fem,2))
println("||A*ones||: ", norm(null_residual))
println("A*ones[1:5]: ", null_residual[1:5])

# Compare with the correct solution behavior
println("\nDirect solver behavior:")
x_direct = A_fem \ b_fem
println("Direct solution[1:5]: ", x_direct[1:5])
println("||A*x_direct - b||: ", norm(A_fem * x_direct - b_fem))

# Check what happens if we solve A*x = A*ones with a different initial guess
println("\nTesting with zero initial guess:")
y_zeros = zeros(size(A_fem,2))
p_zeros = PS.linear_problem(y_zeros, A_fem, b_fem)
s_zeros = PS.minres(p_zeros; verbose=false, reltol=1e-12, iterations=200)
PS.solve(s_zeros)
println("MINRES from zero initial guess[1:5]: ", PS.solution(p_zeros)[1:5])
println("Error: ", norm(x_true - PS.solution(p_zeros)))
