using JLD2, FileIO, ArgParse
using Printf

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--inputpath"
            arg_type=String
            default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/h5path"
        "--outputpath"
            arg_type=String
            default="/home/xiucheng/data-backup/bigtable/2015-taxi/data/jldpath"
    end
    parse_args(s; as_symbols=true)
end

@everywhere include("Trip.jl")
# include("Trip.jl")

function attachroads_fmm!(trips::Vector{Trip})
    """
    Attaching the roads field for each trip.
    """
#     @printf("matching trips...\n")
    println("Total number of trips: " * string(length(trips)))
    @time results = pmap(trip2roads_fmm, trips)
    count = 0
    for i = 1:length(results)
        trips[i].roads = get(results[i], "cpath", -1)
        trips[i].opath = get(results[i], "opath", -1)
        trips[i].mgeom_wkt = get(results[i], "mgeom_wkt", -1)
        trips[i].pgeom_wkt = get(results[i], "pgeom_wkt", -1)
        trips[i].indices = get(results[i], "indices", -1)
        trips[i].offset = get(results[i], "offset", -1)
        trips[i].spdist = get(results[i], "spdist", -1)
        if length(trips[i].roads) == 0
            count+=1
        end
    end
    println("Total number of trips after matching: " * string(length(results)))
    println("Number of trips not matched: " * string(count))
    trips
end

function attachroads!(trips::Vector{Trip})
    """
    Attaching the roads field for each trip.
    """
    println("matching trips...")
    count=0
    @time results = pmap(trip2roads, trips)
    for i = 1:length(results)
        trips[i].roads = results[i]
        if length(trips[i].roads) == 0
            count+=1
        end
    end
    println("Total number of trips: " * string(length(results)))
    println("Number of trips not matched: " * string(count))
    trips
end

function h5f2jld(h5path::String, jldpath::String)
    """
    Attaching road id to the trajectories in h5file and then save them into
    jldfile.
    """
    function h5f2jld(h5file::String)
        println("reading trips from $(basename(h5file))...")
        trips = readtripsh5(h5file)
        trips = filter(isvalidtrip, trips)
        attachroads_fmm!(trips)
        jldfile = basename(h5file) |> splitext |> first |> x->"$x.jld2"
        # save(joinpath(jldpath, jldfile), "trips", trips)
        jldopen(joinpath(jldpath, jldfile), true, true, true, IOStream) do file
            write(file, "trips", trips)
        end
    end

    fnames = filter(x -> endswith(x, ".h5"), readdir(h5path))
    fnames = map(x -> joinpath(h5path, x), fnames)
    for fname in fnames
        h5f2jld(fname)
    end
end



isdir(args[:inputpath]) || error("Invalid inputpath: $(args[:inputpath])")
isdir(args[:outputpath]) || error("Invalid outputpath: $(args[:outputpath])")
h5f2jld(args[:inputpath], args[:outputpath])
