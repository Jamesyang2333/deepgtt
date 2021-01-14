using HDF5, JSON, JLD2, FileIO, Dates, Printf
# include("Link.jl")

# link = Link(1)

test = zeros(Float32, 2, 2)
test[1, 1] = 1
test[1, 2] = 2
test[2, 1] = 3
test[2, 2] = 4

h5open("./test_matrix.h5", "w") do f
    f["tmap"] = test
end