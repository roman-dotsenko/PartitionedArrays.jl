using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS

# Check the FEM Laplacian system
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_fem_true = ones(size(A_fem,2))
b_fem = A_fem * x_fem_true

println("Matrix size: ", size(A_fem))
println("Matrix type: ", typeof(A_fem))
println("Matrix condition number: ", cond(Matrix(A_fem)))
println("Matrix rank: ", rank(Matrix(A_fem)))
println("Matrix determinant: ", det(Matrix(A_fem)))

# Check if A*x = b actually holds
residual_check = A_fem * x_fem_true - b_fem
println("||A*x - b|| = ", norm(residual_check))
println("Should be zero for exact solution")

# Try solving with Julia's built-in solver
x_direct = Matrix(A_fem) \ b_fem
println("Direct solver solution error: ", norm(x_fem_true - x_direct))
println("Expected solution: ", x_fem_true[1:5])
println("Direct solution: ", x_direct[1:5])

# Check if the matrix is singular or ill-conditioned
eigenvalues = LinearAlgebra.eigvals(Matrix(A_fem))
println("Smallest eigenvalue: ", minimum(eigenvalues))
println("Largest eigenvalue: ", maximum(eigenvalues))
println("Condition number: ", maximum(eigenvalues) / minimum(eigenvalues))
