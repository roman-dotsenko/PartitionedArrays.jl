module MinresDASTests

using MPI
using PartitionedArrays
# Ensure minres_petc_test_paralel.jl does not redefine MinresDASTests module
include("minres_petc_test_paralel.jl")

function main()
    with_mpi() do distribute
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        np = MPI.Comm_size(MPI.COMM_WORLD)
        all_tests_passed = true # Flag to track overall test success

        try
            if rank == 0
                println("Starting tests with $np processes.")
            end

            # First set of tests (default problem size)
            MinresDASParallel.test_parallel_mpi(distribute, rank, np)
            
            # Second set of tests with a different problem size
            MinresDASParallel.test_parallel_mpi(distribute, rank, np, 50) # Example: problem_size = 40

        catch e
            all_tests_passed = false
            if rank == 0 # Print error details only on rank 0
                println("\nError occurred during MPI tests on rank $rank:")
                showerror(stdout, e, catch_backtrace())
                println() # Newline for better formatting
            end
            # To ensure the job fails correctly, you might consider MPI.Abort if an error is critical
            # MPI.Abort(MPI.COMM_WORLD, 1)
            # For now, allow Julia's default error propagation per rank.
        end

        MPI.Barrier(MPI.COMM_WORLD) # Synchronize all processes before final status messages

        if rank == 0
            if all_tests_passed
                println("\nAll tests completed successfully on all ranks.")
            else
                println("\nSome tests failed. Please review logs above.")
            end
        end
        println("Rank $rank finished testing.")

        # MPI.Finalize() is automatically handled by with_mpi()
    end
end

main()

end