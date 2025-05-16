module KrylovSolversTests

using Test
import PartitionedSolvers as PS
import PartitionedArrays as PA
import IterativeSolvers
import LinearAlgebra

#tol = 1e-8
## Simple MINRES test
#n = 10
#nodes_per_dir = (n,n,n)
#args = PA.laplacian_fem(nodes_per_dir)
#A = PA.sparse_matrix(args...)
#x = ones(size(A,2))
##A = randn(100,100);
##A = A + A';  # symmetric test matrix
##x = ones(size(A,2))
#b = randn(1000);
#x0, r = PS.minres_solve!(A, b, x, tol)
#@test LinearAlgebra.norm(A*x0-b) <= tol


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
s = PS.minres(p;verbose=true, Pl=Pl)
@time s = PS.solve(s)
@test x ≈ PS.solution(p)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
@test x ≈ 2*PS.solution(p)


#n = 10
#nodes_per_dir = (n,n,n)
#args = PA.laplacian_fem(nodes_per_dir)
#A = PA.sparse_matrix(args...)
#x = ones(size(A,2))
#b = A*x
#
#y = similar(x)
#y .= 0
#p = PS.linear_problem(y,A,b)
#Pl = PS.preconditioner(PS.amg,p)
#s = PS.minres(p;verbose=true, Pl=Pl)
#s = PS.solve(s)
#@test x ≈ PS.solution(p)
#s = PS.update(s,matrix=2*A)
#s = PS.solve(s)
#@test x ≈ 2*PS.solution(p)

#z = IterativeSolvers.cg(A,b;verbose=true,Pl)

parts_per_dir = (2,2,2)
ranks = PA.DebugArray(LinearIndices((prod(parts_per_dir),)))

args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
A = PA.psparse(args...) |> fetch
x = PA.pones(axes(A,2))
b = A*x

y = similar(x)
y .= 0
p = PS.linear_problem(y,A,b)
Pl = PS.preconditioner(PS.amg,p)
s = PS.minres(p;verbose=true,Pl)
@time s = PS.solve(s)
@test x ≈ PS.solution(p)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
@test x ≈ 2*PS.solution(p)

end # module
