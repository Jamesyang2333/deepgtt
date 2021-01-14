using JLD2, FileIO, ArgParse
using Printf

include("Trip.jl")

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

function h5f_sample(h5path_input::String, h5path_output::String)
    """
    sample the first 100 trips
    """
    function h5f_sample(h5file::String)
        println("reading trips from $(basename(h5file))...")
        trips = readtripsh5(h5file)
        sub_trips = trips[1:100]
        h5file_output = h5path_output * ("/") * basename(h5file)
        h5open(h5file_output, "w") do f
            f["/meta/ntrips"] = 100
            for i = 1:length(sub_trips)
                f["/trip/$i/lon"] = sub_trips[i].lon
                f["/trip/$i/lat"] = sub_trips[i].lat
                f["/trip/$i/tms"] = sub_trips[i].tms
            end
        end
    end

    fnames = filter(x -> endswith(x, ".h5"), readdir(h5path_input))
    fnames = map(x -> joinpath(h5path_input, x), fnames)
    for fname in fnames
        h5f_sample(fname)
    end
end



isdir(args[:inputpath]) || error("Invalid inputpath: $(args[:inputpath])")
isdir(args[:outputpath]) || error("Invalid outputpath: $(args[:outputpath])")
h5f_sample(args[:inputpath], args[:outputpath])