module MinresSolverTests

using Test
using BenchmarkTools
import PartitionedSolvers as PS
import PartitionedArrays as PA
import IterativeSolvers
import LinearAlgebra

tol = 1e-7

## Comparison of the PartitionedSolvers.minres with interative minres, uncomment prints to test. The prints should be simmilar.
# Simple MINRES test
#n = 10
#nodes_per_dir = (n,n,n)
#args = PA.laplacian_fem(nodes_per_dir)
#A = PA.sparse_matrix(args...)
#x = ones(size(A,2))
#A1 = randn(20,20);
#A1 = A1 + A1';  # symmetric test matrix
#x1 = ones(size(A1,2))
#b1 = randn(20);
#A2 = copy(A1)
#x2 = copy(x1)
#b2 = copy(b1)
#
#x0, r = PS.minres_solve!(A1, b1, x1, tol)
#@test LinearAlgebra.norm(A1*x0-b1) <= tol
#
#y = copy(x2) # Initialize y with a defined initial guess
#p = PS.linear_problem(y,A2,b2)
#s = PS.minres(p)
#timings = [];
#bch = @benchmark PS.solve(s) # PS.solution(p) is y, which is updated by solve(s)
#display(bch)
#push!(timings, bch)
#
#@test LinearAlgebra.norm(A2 * PS.solution(p) - b2) <= tol # Check residual norm

n = 10
nodes_per_dir = (n,n,n)
args = PA.laplacian_fem(nodes_per_dir)
A = PA.sparse_matrix(args...)
x = ones(size(A,2))
b = A*x
y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
Pl = PS.preconditioner(PS.amg,p)

timings = []
s = PS.minres(p;verbose=true, Pl=Pl, reltol=tol)  # Pass tolerance to solver
@time PS.solve(s)
#display(bch)
#push!(timings, bch)

println("Final solution error: ", LinearAlgebra.norm(x - PS.solution(p)))
println("Solution tolerance requested: ", tol)

@test isapprox(x, PS.solution(p), atol=tol)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
#@test isapprox(x, 2*PS.solution(p), atol=tol)
#
##n = 10
##nodes_per_dir = (n,n,n)
##args = PA.laplacian_fem(nodes_per_dir)
##A = PA.sparse_matrix(args...)
##x = ones(size(A,2))
##b = A*x
##
##y = similar(x)
##y .= 0
##p = PS.linear_problem(y,A,b)
##Pl = PS.preconditioner(PS.amg,p)
##s = PS.minres(p;verbose=true, Pl=Pl)
##s = PS.solve(s)
##@test x ≈ PS.solution(p)
##s = PS.update(s,matrix=2*A)
##s = PS.solve(s)
##@test x ≈ 2*PS.solution(p)
#
##z = IterativeSolvers.cg(A,b;verbose=true,Pl)
#
#parts_per_dir = (2,2,2)
#ranks = PA.DebugArray(LinearIndices((prod(parts_per_dir),)))
#
#args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
#A = PA.psparse(args...) |> fetch
#x = PA.pones(axes(A,2))
#b = A*x
#
#y = similar(x)
#y .= 0
#p = PS.linear_problem(y,A,b)
#Pl = PS.preconditioner(PS.amg,p)
#s = PS.minres(p;verbose=false,Pl)
#@time s = PS.solve(s)
#@test x ≈ PS.solution(p)
#s = PS.update(s,matrix=2*A)
#s = PS.solve(s)
#@test x ≈ 2*PS.solution(p)
##
end # module
