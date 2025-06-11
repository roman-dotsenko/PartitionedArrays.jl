# Working folder
It is always expected and required to work in the PartitionedSolvers directory. If you are not in the PartitionedSolvers directory do `cd ./PartitionedSolvers`
# Running the tests for the minres:
It is possible to run tests using the command `julia --project=. -e "include("PartitionedSolvers/test/minres_petc_tests.jl")`. The minres_petc_tests.jl file as an example.
# Working boundaries
The work should be done only inside PartitionedSolvers folder, and in no circumstance it is needed to change anything outside PartitionedSolvers unless it is environment setup. Installation of the julia packages needs to be done only in the PartitionedSolvers project.