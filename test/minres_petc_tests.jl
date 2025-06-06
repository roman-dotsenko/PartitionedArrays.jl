using SparseArrays
using Test
using PetscCall
import PartitionedSolvers as PS
import PartitionedArrays as PA
using LinearAlgebra
using BenchmarkTools
import IterativeSolvers


function test_pa_minres(p_problem, Pl_preconditioner, rank, A)
    s = PS.minres(p_problem;verbose=true, Pl=Pl_preconditioner)
    @time PS.solve(s)
        #display(bch)
    x = ones(size(A,2))
    #@test isapprox(x, PS.solution(p_problem), atol=1e-5)
end

function test_petsc_minres(matrix_A, x, vector_b, verbose)
    # PetscCall options -ksp_converged_reason

    options = "-ksp_type minres -pc_type none -ksp_rtol 3.6e-8 -ksp_monitor"
    if verbose === true
        options += " -ksp_monitor"
    end
    
    PetscCall.init(args=split(options))

    # Say, we want to solve A*x=b
    x2 = similar(x); x2 .= 0
    setup = PetscCall.ksp_setup(x2,matrix_A,vector_b)
    if verbose === true
        results = PetscCall.ksp_solve!(x2,setup,vector_b)
    else
        @time PetscCall.ksp_solve!(x2,setup,vector_b)
        #display(bench)
    end
    # The user needs to explicitly destroy
    # the setup object. This cannot be hidden in
    # Julia finalizers since destructors in petsc are
    # collective operations (in parallel runs).
    # Julia finalizers do not guarantee this.
    PetscCall.ksp_finalize!(setup)
    @test isapprox(x, x2, atol=1e-4)
end

function test_paralel_pa_minres(problem, A, x, verbose)
    #Pl = PS.preconditioner(PS.amg,p)
    s = PS.minres(problem;verbose=verbose)
    @time s = PS.solve(s)
    @test isapprox(x, PS.solution(problem), atol=1e-5)
    s = PS.update(s,matrix=2*A)
    s = PS.solve(s)
    @test isapprox(x, 2*PS.solution(problem), atol=1e-5)
end


function test_linear()
    n = 40
    nodes_per_dir = (n,n,n)
    args = PA.laplacian_fem(nodes_per_dir)
    A = PA.sparse_matrix(args...)
    x = ones(size(A,2))
    b = A*x
    y = similar(x)
    y .= 0
    p = PS.linear_problem(y,A,b)
    Pl = PS.preconditioner(PS.amg,p)
    println("Testing PA minres with $(size(A,2)) rows\n")
    test_pa_minres(p, Pl, false, A)
    println("Testing PETSc minres with  $(size(A,2)) rows\n")
    test_petsc_minres(A, x, b, false)
end

function test_paralel()
    n = 10
    nodes_per_dir = (n,n,n)
    parts_per_dir = (2,2,2)
    ranks = PA.DebugArray(LinearIndices((prod(parts_per_dir),)))

    args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
    A = PA.psparse(args...) |> fetch
    x = PA.pones(axes(A,2))
    b = A*x

    y = similar(x)
    y .= 0
    p = PS.linear_problem(y,A,b)

    println("Testing paralel PA minres with size $(size(A,2))\n")
    test_paralel_pa_minres(p, A, x, false)
    
    println("Testing paralel PETSc minres with size $(size(A,2))\n")
end


test_linear()
test_linear()

#z = IterativeSolvers.minres!(x,A,b;verbose=true)


## Create a spares matrix and a vector in Julia
#I = [1,1,2,2,2,3,3,3,4,4]
#J = [1,2,1,2,3,2,3,4,3,4]
#V = [4,-2,-1,6,-2,-1,6,-2,-1,4]
#m = 4
#n = 4
#A = sparse(I,J,V,m,n)
#x = ones(m)
#b = A*x

# PetscCall options -ksp_converged_reason
#options = "-ksp_type minres -pc_type ilu -log_omit -log_view :skip"
#PetscCall.init(args=split(options))
#
## Say, we want to solve A*x=b
#x2 = similar(x); x2 .= 0
#setup = PetscCall.ksp_setup(x2,A,b)
##results = PetscCall.ksp_solve!(x2,setup,b)
#bench = @benchmark PetscCall.ksp_solve!(x2,setup,b)
#display(bench)
#
#@test x ≈ x2
#
## Info about the solution process
#@show results
#
## Now with the same matrix, but a different rhs
#b = 2*b
#results = PetscCall.ksp_solve!(x2,setup,b)
#@test 2*x ≈ x2
#
## Now with a different matrix, but reusing as much as possible
## from the previous solve.
#A = 2*A
#PetscCall.ksp_setup!(setup,A)
#results = PetscCall.ksp_solve!(x2,setup,b)
#@test x ≈ x2
#

# The setup object cannot be used anymore.
# This now would be provably a code dump:
# PetscCall.ksp_solve!(x2,setup,b)