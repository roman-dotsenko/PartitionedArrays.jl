module KrylovSolversTests

using Test
import PartitionedSolvers as PS
import PartitionedArrays as PA
import IterativeSolvers

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
s = PS.pcg(p;verbose=true,Pl)
s = PS.solve(s)
@test x ≈ PS.solution(p)
s = PS.update(s,matrix=2*A)
s = PS.solve(s)
@test x ≈ 2*PS.solution(p)

z = IterativeSolvers.cg(A,b;verbose=true,Pl)



end # module
