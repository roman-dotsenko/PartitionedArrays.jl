using LinearAlgebra
import PartitionedArrays as PA
import PartitionedSolvers as PS
import IterativeSolvers

println("=== Comparing Our MINRES vs IterativeSolvers MINRES ===")

# Create FEM Laplacian problem
n = 5
args = PA.laplacian_fem((n,n,n))
A_fem = PA.sparse_matrix(args...)
x_true = ones(size(A_fem,2))
b_fem = A_fem * x_true

println("Matrix size: ", size(A_fem))
println("True solution: ", x_true[1:5])

# Test IterativeSolvers MINRES
println("\n--- IterativeSolvers MINRES ---")
x_iter = zeros(size(A_fem,2))
result_iter = IterativeSolvers.minres!(x_iter, A_fem, b_fem, reltol=1e-12, maxiter=10, log=true)
println("IterativeSolvers converged: ", result_iter[2].isconverged)
println("IterativeSolvers iterations: ", result_iter[2].iters)
println("IterativeSolvers solution: ", x_iter[1:5])
println("IterativeSolvers error: ", norm(x_iter - x_true))
println("IterativeSolvers residual: ", norm(A_fem * x_iter - b_fem))

# Test our MINRES
println("\n--- Our MINRES ---")
y = zeros(size(A_fem,2))
p = PS.linear_problem(y, A_fem, b_fem)
s = PS.minres(p; verbose=false, reltol=1e-12, iterations=10)
PS.solve(s)
println("Our MINRES solution: ", PS.solution(p)[1:5])
println("Our MINRES error: ", norm(PS.solution(p) - x_true))
println("Our MINRES residual: ", norm(A_fem * PS.solution(p) - b_fem))

# Compare the solutions
println("\n--- Comparison ---")
println("Solution difference: ", norm(x_iter - PS.solution(p)))
println("Error ratio (ours/theirs): ", norm(PS.solution(p) - x_true) / norm(x_iter - x_true))
