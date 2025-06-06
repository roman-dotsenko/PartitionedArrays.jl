module MinresDASParallel

using SparseArrays
using Test
using PetscCall
import PartitionedSolvers as PS
import PartitionedArrays as PA
using LinearAlgebra
using BenchmarkTools
using MPI
import IterativeSolvers
using ProfileView

function test_parallel_pa_minres(problem, rank)
    #Pl = PS.preconditioner(PS.amg,p)
    s = PS.minres(problem; verbose=false) # Or your minres setup

    if rank == 0
        @time s_solved = PS.solve(s)
    else
        s_solved = PS.solve(s)
    end

    return PS.solution(problem) # Ensure the solution is computed
end

function test_parallel(distribute, np)
    n = 40
    nodes_per_dir = (n,n,n)
    parts_per_dir = (np,np,np)
    ranks = PA.DebugArray(LinearIndices((prod(parts_per_dir),)))

    args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
    A = PA.psparse(args...) |> fetch
    x = PA.pones(axes(A,2))
    b = A*x

    y = similar(x)
    y .= 0
    p = PS.linear_problem(y,A,b)

    println("Testing parallel PA minres with size $(size(A,2))\n")
    test_parallel_pa_minres(p, A, x, false)
    
    println("Testing paralel PETSc minres with size $(size(A,2))\n")
end

function calculate_parts_per_dir(np)
    # Find approximate cube root
    cube_root = round(Int, np^(1/3))
    
    # Start with cube_root in each direction
    nx = ny = nz = cube_root
    
    # Adjust to use all processes
    remaining = np ÷ (nx * ny * nz)
    
    # Distribute remaining processes by increasing dimensions
    while nx * ny * nz < np
        if nx <= ny && nx <= nz
            nx += 1
        elseif ny <= nz
            ny += 1
        else
            nz += 1
        end
    end
    
    # If we went over, reduce the largest dimension
    while nx * ny * nz > np
        if nx >= ny && nx >= nz
            nx -= 1
        elseif ny >= nz
            ny -= 1
        else
            nz -= 1
        end
    end
    
    return (nx, ny, nz)
end

function test_parallel_mpi(distribute, rank, np, problem_size=10)
    n = problem_size
    nodes_per_dir = (n,n,n)
    parts_per_dir = calculate_parts_per_dir(np)

    total_parts = prod(parts_per_dir)
    ranks = distribute(LinearIndices((total_parts,)))

    args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
    A = PA.psparse(args...) |> fetch
    x = PA.pones(axes(A,2))
    b = A*x

    y = similar(x)
    y .= 0
    p = PS.linear_problem(y,A,b)

    if rank == 0
        println("Testing parallel PA minres with size $(size(A,2))\n")
    end

    computed_solution = test_parallel_pa_minres(p, rank)
    
    MPI.Barrier(MPI.COMM_WORLD)

    if rank == 0
        println("Attempting isapprox comparison...")
    end

    comparison_result = false # Default to false
    try
        comparison_result = isapprox(x, computed_solution, atol=1e-5)
        if rank == 0
            println("isapprox result: $comparison_result")
        end
        # isapprox for PVectors should return the same boolean on all ranks
        if rank == 0
            println("isapprox completed.")
        end
    catch e
        if rank == 0
            println("Error during isapprox: $e")
        end
        # Consider test failed
    end


    if rank == 0
        println("Testing parallel PETSc minres with size $(size(A,2))\n")
    end

    n = problem_size
    nodes_per_dir = (n,n,n)
    parts_per_dir = calculate_parts_per_dir(np)

    total_parts = prod(parts_per_dir)
    ranks = PA.DebugArray(LinearIndices((total_parts,)))

    args = PA.laplacian_fem(nodes_per_dir,parts_per_dir,ranks)
    A = PA.psparse(args...) |> fetch
    x = PA.pones(axes(A,2))
    b = A*x

    y = similar(x)
    y .= 0
    p = PS.linear_problem(y,A,b)

    options = "-ksp_type minres -pc_type none -ksp_rtol 3.6e-8"
    if rank === 0
        options *= " -ksp_monitor"
    end
    
    PetscCall.init(args=split(options))

    # Say, we want to solve A*x=b
    x2 = similar(x); x2 .= 0
    setup = PetscCall.ksp_setup(x2,A,b)
    if rank != 0 
        results = PetscCall.ksp_solve!(x2,setup,b)
    else
        @time PetscCall.ksp_solve!(x2,setup,b)
        #display(bench)
    end

    # The user needs to explicitly destroy
    # the setup object. This cannot be hidden in
    # Julia finalizers since destructors in petsc are
    # collective operations (in parallel runs).
    # Julia finalizers do not guarantee this.
    PetscCall.ksp_finalize!(setup)
end

#test_paralel()
#test_paralel()

end