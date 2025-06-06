# run_tests.slurm
#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=julia_mpi_minres_tests

# --- Load required modules ---
echo "Loading modules..."

module is-loaded openmpi4 || module load openmpi4/4.1.6
module is-loaded julia || module load julia/1.10.3 > /dev/null

# --- Create julia environment ---

if [ ! -d "Project.toml" ]; then 
    echo "Creating Julia project environment..."
    julia --project=. -e 'using Pkg; Pkg.activate(".")'
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
else
    echo "Julia project environment already exists."
fi

# --- Add julia packages ---
julia --project=. -e 'using Pkg; Pkg.add("MPI")'
julia --project=. -e 'using Pkg; Pkg.add("MPIPreferences")'
julia --project=. -e 'using Pkg; Pkg.add("BenchmarkTools")'
julia --project=. -e 'using Pkg; Pkg.add("IterativeSolvers")'
julia --project=. -e 'using Pkg; Pkg.add("NLsolve")'
julia --project=. -e 'using Pkg; Pkg.add("PartitionedArrays")'
julia --project=. -e 'using Pkg; Pkg.add("PetscCall")'
julia --project=. -e 'using Pkg; Pkg.add("SparseArrays")'
julia --project=. -e 'using Pkg; Pkg.add("SparseMatricesCSR")'

# --- Precompiling Julia packages ---
echo "Precompiling Julia packages..."
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia --project=. -O3 --check-bounds=no -e 'using Pkg; Pkg.add("SparseMatricesCSR")'

echo "✅ Julia MPI environment is successfuly set up!"
echo "Running tests..."
julia --project=. -e 'include("./test/minres_das_tests.jl")'
